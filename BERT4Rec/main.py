

import torch, math
import torch.nn as nn

import freerec
from freerec.data.fields import FieldModuleList
from freerec.data.tags import USER, SESSION, ITEM, TIMESTAMP, ID, POSITIVE, UNSEEN, SEEN

from sampler import *

freerec.declare(version='0.6.3')

cfg = freerec.parser.Parser()

# BERT4Rec
cfg.add_argument("--maxlen", type=int, default=20)
cfg.add_argument("--num-heads", type=int, default=4)
cfg.add_argument("--num-blocks", type=int, default=2)
cfg.add_argument("--hidden-size", type=int, default=64)
cfg.add_argument("--dropout-rate", type=float, default=0.2)
cfg.add_argument("--mask-prob", type=float, default=0.2, help="the probability of masking")

cfg.add_argument("--decay-step", type=int, default=25)
cfg.add_argument("--decay-factor", type=float, default=1., help="lr *= factor per decay step")

# Loss
cfg.add_argument(
    "--loss", 
    choices=(
        "BCE", "BPR", "CE", "CE-Sampled", "SCE", "CE-Tight", "NCE", "NEG",
        "bce", "bpr", "ce", "ce-sampled", "sce", "ce-tight", "nce", "neg"
    ), 
    default="SCE"
)
cfg.add_argument("--c", type=float, default=None, help="the hyper-parameter for NCE")
cfg.add_argument("--alpha", type=float, default=None, help="the hyper-parameter for Scaled loss")
cfg.add_argument("--eta", type=float, default=None, help="the hyper-parameter for CE-Tight")
cfg.add_argument("--num-negs", type=int, default=1, help="the number of negative samples")
cfg.add_argument("--neg-pool", choices=("unseen", "non-target", "all"), default='non-target', help="the pool from which the negatives are sampled")

cfg.set_defaults(
    description="BERT4Rec",
    root="../../data",
    dataset='MovieLens1M_550_Chron',
    epochs=100,
    batch_size=256,
    optimizer='adam',
    lr=1e-3,
    weight_decay=0.,
    seed=1,
)
cfg.compile()


NUM_PADS = 2


class BERT4Rec(freerec.models.RecSysArch):

    def __init__(
        self, fields: FieldModuleList,
        maxlen: int = cfg.maxlen,
        hidden_size: int = cfg.hidden_size,
        dropout_rate: float = cfg.dropout_rate,
        num_blocks: int = cfg.num_blocks,
        num_heads: int = cfg.num_heads,
    ) -> None:
        super().__init__()

        self.num_blocks = num_blocks
        self.fields = fields
        self.Item = self.fields[ITEM, ID]

        self.Position = nn.Embedding(maxlen, hidden_size)
        self.embdDropout = nn.Dropout(p=dropout_rate)
        self.register_buffer(
            "positions_ids",
            torch.tensor(range(0, maxlen), dtype=torch.long).unsqueeze(0)
        )

        self.layernorm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout_rate)
        self.encoder = nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(
                d_model=hidden_size,
                nhead=num_heads,
                dim_feedforward=hidden_size * 4,
                dropout=dropout_rate,
                activation='gelu',
                batch_first=True
            ),
            num_layers=num_blocks
        )

        self.reset_parameters()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                m.weight.data.clamp_(-0.02, 0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.)
            elif isinstance(m, nn.Embedding):
                nn.init.xavier_normal_(m.weight)
                m.weight.data.clamp_(-0.02, 0.02)

    def mark_position(self, seqs: torch.Tensor):
        positions = self.Position(self.positions_ids) # (1, maxlen, D)
        return seqs + positions

    def forward(self, seqs: torch.Tensor):
        padding_mask = seqs == 0
        seqs = self.mark_position(self.Item.look_up(seqs)) # (B, S) -> (B, S, D)
        seqs = self.dropout(self.layernorm(seqs))

        features = self.encoder(seqs, src_key_padding_mask=padding_mask) # (B, S, D)
        return features

    def predict(
        self, 
        seqs: torch.Tensor,
        positives: torch.Tensor,
        negatives: torch.Tensor
    ):
        features = self.forward(seqs) # (B, S, D)
        posEmbds = self.Item.look_up(positives) # (B, S, D)
        negEmbds = self.Item.look_up(negatives) # (B, S, K, D)

        posLogits = features.mul(posEmbds).sum(-1, keepdim=True) # (B, S, 1)
        negLogits = None
        logits = None
        # Sometimes CE is faster because:
        # https://github.com/pytorch/pytorch/issues/113934
        if cfg.loss.lower() in ('ce', 'ce-tight'):
            items = self.Item.embeddings.weight[NUM_PADS:] # (N, D)
            logits = features.matmul(items.t()) # (B, S, N)
        else:
            negLogits = negEmbds.matmul(features.unsqueeze(-1)).squeeze(-1) # (B, S, K)
        return posLogits, negLogits, logits

    def recommend_from_pool(self, seqs: torch.Tensor, pool: torch.Tensor):
        features = self.forward(seqs)[:, [-1], :]  # (B, 1, D)
        others = self.Item.look_up(pool) # (B, K, D)
        return features.mul(others).sum(-1)

    def recommend_from_full(self, seqs: torch.Tensor):
        features = self.forward(seqs)[:, -1, :]  # (B, D)
        items = self.Item.embeddings.weight[NUM_PADS:] # (N, D)
        return features.matmul(items.t()) # (B, N)


class CoachForBERT4Rec(freerec.launcher.SeqCoach):

    def random_mask(self, seqs: torch.Tensor, p: float = cfg.mask_prob):
        padding_mask = seqs == 0
        rnds = torch.rand(seqs.size(), device=seqs.device)
        masked_seqs = torch.where(rnds < p, torch.ones_like(seqs), seqs)
        masked_seqs.masked_fill_(padding_mask, 0)
        masks = (masked_seqs == 1) # (B, S)
        return masked_seqs, masks

    def sample_negs(self, negatives: torch.Tensor, count: int):
        if self.cfg.neg_pool == 'all':
            B, S = negatives.shape[:2]
            return torch.randint(
                NUM_PADS, count + NUM_PADS, 
                (B, S, self.cfg.num_negs),
                device=self.device
            )
        else:
            return negatives

    def train_per_epoch(self, epoch: int):
        Item = self.fields[ITEM, ID]
        for data in self.dataloader:
            users, seqs, negatives = [col.to(self.device) for col in data]
            positives = seqs
            masked_seqs, indices = self.random_mask(seqs)
            negatives = self.sample_negs(negatives, Item.count)
            posLogits, negLogits, logits = self.model.predict(masked_seqs, seqs, negatives)
            if cfg.loss.lower() == 'bce':
                posLogits = posLogits[indices] # (*, 1)
                negLogits = negLogits[indices] # (*, K)
                posLabels = torch.ones_like(posLogits)
                negLabels = torch.zeros_like(negLogits)
                loss = self.criterion(posLogits, posLabels) \
                    + self.criterion(negLogits, negLabels)
            elif cfg.loss.lower() == 'bpr':
                posLogits = posLogits[indices] # (*, 1)
                negLogits = negLogits[indices] # (*, K)
                loss = self.criterion(posLogits, negLogits)
            elif cfg.loss.lower() == 'ce':
                logits = logits[indices] # (*, N)
                targets = positives[indices].flatten() # (*,)
                loss = self.criterion(logits, targets - NUM_PADS)
            elif cfg.loss.lower() == 'ce-tight':
                logits = logits[indices] # (*, N)
                targets = positives[indices].flatten() - NUM_PADS # (*,)
                posLogits = torch.gather(logits, 1, targets.unsqueeze(-1)).detach()
                logits = logits - posLogits
                logits = torch.where(
                    logits < cfg.eta * posLogits.abs().neg() - 1e-8,
                    torch.empty_like(logits).fill_(-1e23),
                    logits
                )
                loss = self.criterion(logits, targets)
            elif cfg.loss.lower() == 'ce-sampled':
                logits = torch.cat((posLogits, negLogits), dim=-1) # (B, S, K + 1)
                logits = logits[indices] # (*, K + 1)
                targets = torch.zeros((len(logits),), device=self.device, dtype=torch.long) # (*,)
                loss = self.criterion(logits, targets)
            elif cfg.loss.lower() == 'sce':
                negLogits = negLogits + math.log(cfg.alpha)
                logits = torch.cat((posLogits, negLogits), dim=-1) # (B, S, K + 1)
                logits = logits[indices] # (*, K + 1)
                targets = torch.zeros((len(logits),), device=self.device, dtype=torch.long) # (*,)
                loss = self.criterion(logits, targets)
            elif cfg.loss.lower() == 'nce':
                C = math.log((Item.count) / cfg.num_negs)
                posLogits = posLogits[indices] + C - cfg.c # (*, 1)
                negLogits = negLogits[indices] + C - cfg.c # (*, K)
                posLabels = torch.ones_like(posLogits)
                negLabels = torch.zeros_like(negLogits)
                loss = self.criterion(posLogits, posLabels) \
                    + self.criterion(negLogits, negLabels) * cfg.num_negs
            elif cfg.loss.lower() == 'neg':
                posLogits = posLogits[indices] # (*, 1)
                negLogits = negLogits[indices] # (*, K)
                posLabels = torch.ones_like(posLogits)
                negLabels = torch.zeros_like(negLogits)
                loss = self.criterion(posLogits, posLabels) \
                    + self.criterion(negLogits, negLabels) * cfg.num_negs

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            self.monitor(loss.item(), n=users.size(0), mode="mean", prefix='train', pool=['LOSS'])


def main():

    dataset = getattr(freerec.data.datasets.sequential, cfg.dataset)(root=cfg.root)
    User, Item = dataset.fields[USER, ID], dataset.fields[ITEM, ID]

    if cfg.neg_pool == 'all':
        # trainpipe
        trainpipe = freerec.data.postprocessing.source.RandomShuffledSource(
            source=dataset.train().to_seqs(keepid=True)
        ).sharding_filter().seq_train_sampling_(
            dataset, leave_one_out=False, 
            num_negatives=1, pool=cfg.neg_pool # yielding (user, seqs, negatives)
        ).lprune_(
            indices=[1, 2], maxlen=cfg.maxlen
        ).add_(
            indices=[1, 2], offset=NUM_PADS
        ).lpad_(
            indices=[1, 2], maxlen=cfg.maxlen, padding_value=0 # 0: padding; 1: mask token
        ).batch(cfg.batch_size).column_().tensor_()
    else:
        # trainpipe
        trainpipe = freerec.data.postprocessing.source.RandomShuffledSource(
            source=dataset.train().to_seqs(keepid=True)
        ).sharding_filter().seq_train_sampling_(
            dataset, leave_one_out=False, 
            num_negatives=cfg.num_negs, pool=cfg.neg_pool # yielding (user, seqs, negatives)
        ).lprune_(
            indices=[1, 2], maxlen=cfg.maxlen
        ).add_(
            indices=[1, 2], offset=NUM_PADS
        ).lpad_(
            indices=[1, 2], maxlen=cfg.maxlen, padding_value=0 # 0: padding; 1: mask token
        ).batch(cfg.batch_size).column_().tensor_()

    # validpipe
    if cfg.ranking == 'full':
        validpipe = freerec.data.postprocessing.source.OrderedIDs(
            field=User
        ).sharding_filter().seq_valid_yielding_(
            dataset
        ).lprune_(
            indices=[1], maxlen=cfg.maxlen - 1,
        ).add_(
            indices=[1], offset=NUM_PADS
        ).lpad_(
            indices=[1], maxlen=cfg.maxlen - 1, padding_value=0
        ).rpad_(
            indices=[1], maxlen=cfg.maxlen, padding_value=1 # 1: mask token
        ).batch(128).column_().tensor_().field_(
            User.buffer(), Item.buffer(tags=POSITIVE), Item.buffer(tags=UNSEEN), Item.buffer(tags=SEEN)
        )
    elif cfg.ranking == 'pool':
        validpipe = freerec.data.postprocessing.source.OrderedIDs(
            field=User
        ).sharding_filter().seq_valid_sampling_(
            dataset # yielding (user, items, (target + (100) negatives))
        ).lprune_(
            indices=[1], maxlen=cfg.maxlen - 1,
        ).add_(
            indices=[1, 2], offset=NUM_PADS
        ).lpad_(
            indices=[1], maxlen=cfg.maxlen - 1, padding_value=0
        ).rpad_(
            indices=[1], maxlen=cfg.maxlen, padding_value=1 # 1: mask token
        ).batch(128).column_().tensor_()

    # testpipe
    if cfg.ranking == 'full':
        testpipe = freerec.data.postprocessing.source.OrderedIDs(
            field=User
        ).sharding_filter().seq_test_yielding_(
            dataset
        ).lprune_(
            indices=[1], maxlen=cfg.maxlen - 1,
        ).add_(
            indices=[1], offset=NUM_PADS
        ).lpad_(
            indices=[1], maxlen=cfg.maxlen - 1, padding_value=0
        ).rpad_(
            indices=[1], maxlen=cfg.maxlen, padding_value=1 # 1: mask token
        ).batch(100).column_().tensor_().field_(
            User.buffer(), Item.buffer(tags=POSITIVE), Item.buffer(tags=UNSEEN), Item.buffer(tags=SEEN)
        )
    elif cfg.ranking == 'pool':
        testpipe = freerec.data.postprocessing.source.OrderedIDs(
            field=User
        ).sharding_filter().seq_test_sampling_(
            dataset # yielding (user, items, (target + (100) negatives))
        ).lprune_(
            indices=[1], maxlen=cfg.maxlen - 1,
        ).add_(
            indices=[1, 2], offset=NUM_PADS
        ).lpad_(
            indices=[1], maxlen=cfg.maxlen - 1, padding_value=0
        ).rpad_(
            indices=[1], maxlen=cfg.maxlen, padding_value=1 # 1: mask token
        ).batch(cfg.batch_size).column_().tensor_()

    Item.embed(
        cfg.hidden_size, 
        num_embeddings=Item.count + NUM_PADS,
        padding_idx=0
    )
    tokenizer = FieldModuleList(dataset.fields)
    model = BERT4Rec(
        tokenizer
    )

    if cfg.optimizer == 'sgd':
        optimizer = torch.optim.SGD(
            model.parameters(), lr=cfg.lr, 
            momentum=cfg.momentum,
            nesterov=cfg.nesterov,
            weight_decay=cfg.weight_decay
        )
    elif cfg.optimizer == 'adam':
        optimizer = torch.optim.Adam(
            model.parameters(), lr=cfg.lr,
            betas=(cfg.beta1, cfg.beta2),
            weight_decay=cfg.weight_decay
        )
    elif cfg.optimizer == 'adamw':
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=cfg.lr,
            betas=(cfg.beta1, cfg.beta2),
            weight_decay=cfg.weight_decay
        )
    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer=optimizer,
        step_size=cfg.decay_step,
        gamma=cfg.decay_factor
    )
    if cfg.loss.lower() in ('bce', 'nce', 'neg'):
        criterion = freerec.criterions.BCELoss4Logits(reduction='mean')
    elif cfg.loss.lower() == 'bpr':
        criterion = freerec.criterions.BPRLoss(reduction='mean')
    elif cfg.loss.lower() in ('ce', 'ce-sampled', 'sce', 'ce-tight'):
        criterion = freerec.criterions.CrossEntropy4Logits(reduction='mean')
    else:
        raise NotImplementedError(f"{cfg.loss} is not supported ...")

    coach = CoachForBERT4Rec(
        dataset=dataset,
        trainpipe=trainpipe,
        validpipe=validpipe,
        testpipe=testpipe,
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        device=cfg.device
    )
    coach.compile(
        cfg, 
        monitors=[
            'loss', 
            'hitrate@1', 'hitrate@5', 'hitrate@10',
            'ndcg@5', 'ndcg@10',
            'mrr@5', 'mrr@10'
        ],
        which4best='ndcg@10'
    )
    coach.fit()


if __name__ == "__main__":
    main()