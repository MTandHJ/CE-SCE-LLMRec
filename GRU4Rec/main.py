

import torch, math
import torch.nn as nn

import freerec
from freerec.data.fields import FieldModuleList
from freerec.data.tags import USER, SESSION, ITEM, TIMESTAMP, ID

from sampler import *

freerec.declare(version='0.6.3')

# GRU4Rec
cfg = freerec.parser.Parser()
cfg.add_argument("--maxlen", type=int, default=200)
cfg.add_argument("--embedding-dim", type=int, default=50)
cfg.add_argument("--dropout-rate", type=float, default=0.25)
cfg.add_argument("--num-gru-layers", type=int, default=1)

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
    description="GRU4Rec",
    root="../../data",
    dataset='MovieLens1M_550_Chron',
    epochs=30,
    batch_size=512,
    optimizer='adam',
    lr=1e-3,
    weight_decay=1.e-8,
    seed=1,
)
cfg.compile()
cfg.hidden_size = cfg.embedding_dim


NUM_PADS = 1


class GRU4Rec(freerec.models.RecSysArch):

    def __init__(
        self, fields: FieldModuleList,
    ) -> None:
        super().__init__()

        self.fields = fields
        self.Item = self.fields[ITEM, ID]

        self.emb_dropout = nn.Dropout(cfg.dropout_rate)
        self.gru = nn.GRU(
            cfg.embedding_dim,
            cfg.hidden_size,
            cfg.num_gru_layers,
            bias=False,
            batch_first=True,
        )
        self.dense = nn.Linear(cfg.hidden_size, cfg.embedding_dim)

        self.reset_parameters()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Embedding):
                nn.init.xavier_normal_(m.weight)
            elif isinstance(m, nn.GRU):
                nn.init.xavier_uniform_(m.weight_hh_l0)
                nn.init.xavier_uniform_(m.weight_ih_l0)

    def forward(self, seqs: torch.Tensor):
        masks = seqs.not_equal(0).unsqueeze(-1) # (B, S, 1)
        seqs = self.Item.look_up(seqs) # (B, S, D)
        seqs = self.emb_dropout(seqs)
        gru_out, _ = self.gru(seqs) # (B, S, H)

        gru_out = self.dense(gru_out) # (B, S, D)
        features = gru_out.gather(
            dim=1,
            index=masks.sum(1, keepdim=True).add(-1).expand((-1, 1, gru_out.size(-1)))
        ).squeeze(1) # (B, D)

        return features

    def predict(
        self, 
        seqs: torch.Tensor, 
        positives: torch.Tensor, 
        negatives:torch.Tensor
    ):
        features = self.forward(seqs) # (B, D)
        posEmbds = self.Item.look_up(positives) # (B, 1, D)
        negEmbds = self.Item.look_up(negatives) # (B, K, D)

        posLogits = features.unsqueeze(1).mul(posEmbds).sum(-1) # (B, 1)
        negLogits = None
        logits = None
        # Sometimes CE is faster because:
        # https://github.com/pytorch/pytorch/issues/113934
        if cfg.loss.lower() in ('ce', 'ce-tight'):
            items = self.Item.embeddings.weight[NUM_PADS:] # (N, D)
            logits = features.matmul(items.t()) # (B, N)
        else:
            negLogits = negEmbds.matmul(features.unsqueeze(-1)).squeeze(-1) # (B, K)
        return posLogits, negLogits, logits

    def recommend_from_pool(self, seqs: torch.Tensor, pool: torch.Tensor):
        features = self.forward(seqs).unsqueeze(1) # (B, 1, D)
        items = self.Item.look_up(pool) # (B, K, D)
        return features.mul(items).sum(-1)

    def recommend_from_full(self, seqs: torch.Tensor):
        features = self.forward(seqs)
        items = self.Item.embeddings.weight[NUM_PADS:] # (N, D)
        return features.matmul(items.t())


class CoachForGRU4Rec(freerec.launcher.SeqCoach):

    @torch.no_grad()
    def opt_correction(self, negLogits: torch.Tensor, N: int, K: int):
        negLogits_exp = negLogits.exp().detach()
        m = negLogits_exp.mean().square().item()
        s = negLogits_exp.square().mean().item()
        C = (N - 1) * m / (s + (K - 1) * m)
        return C

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
            users, seqs, positives, negatives = [col.to(self.device) for col in data]
            negatives = self.sample_negs(negatives, Item.count)
            posLogits, negLogits, logits = self.model.predict(seqs, positives, negatives.squeeze(1))
            if cfg.loss.lower() == 'bce':
                posLabels = torch.ones_like(posLogits)
                negLabels = torch.zeros_like(negLogits)
                loss = self.criterion(posLogits, posLabels) \
                    + self.criterion(negLogits, negLabels)
            elif cfg.loss.lower() == 'bpr':
                loss = self.criterion(posLogits, negLogits)
            elif cfg.loss.lower() == 'ce':
                targets = positives.flatten()
                loss = self.criterion(logits, targets - NUM_PADS)
            elif cfg.loss.lower() == 'ce-tight':
                targets = positives.flatten() - NUM_PADS # (*,)
                posLogits = torch.gather(logits, 1, targets.unsqueeze(-1)).detach()
                logits = logits - posLogits
                logits = torch.where(
                    logits < cfg.eta * posLogits.abs().neg() - 1e-8,
                    torch.empty_like(logits).fill_(-1e23),
                    logits
                )
                loss = self.criterion(logits, targets)
            elif cfg.loss.lower() == 'ce-sampled':
                logits = torch.cat((posLogits, negLogits), dim=-1) # (B, K + 1)
                targets = torch.zeros((len(logits),), device=self.device, dtype=torch.long) # (B,)
                loss = self.criterion(logits, targets)
            elif cfg.loss.lower() == 'sce':
                negLogits = negLogits + math.log(cfg.alpha)
                logits = torch.cat((posLogits, negLogits), dim=-1) # (B, K + 1)
                targets = torch.zeros((len(logits),), device=self.device, dtype=torch.long) # (B,)
                loss = self.criterion(logits, targets)
            elif cfg.loss.lower() == 'nce':
                C = math.log((Item.count) / cfg.num_negs)
                posLogits = posLogits + C - cfg.c # (*, 1)
                negLogits = negLogits + C - cfg.c # (*, K)
                posLabels = torch.ones_like(posLogits)
                negLabels = torch.zeros_like(negLogits)
                loss = self.criterion(posLogits, posLabels) \
                    + self.criterion(negLogits, negLabels) * cfg.num_negs
            elif cfg.loss.lower() == 'neg':
                posLabels = torch.ones_like(posLogits)
                negLabels = torch.zeros_like(negLogits)
                loss = self.criterion(posLogits, posLabels) \
                    + self.criterion(negLogits, negLabels) * cfg.num_negs

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            self.monitor(loss.item(), n=users.size(0), mode="mean", prefix='train', pool=['LOSS'])


def to_roll_seqs(dataset: freerec.data.datasets.RecDataSet, minlen: int = 2):
    seqs = dataset.to_seqs(keepid=True)

    roll_seqs = []
    for id_, items in seqs:
        for k in range(minlen, len(items) + 1):
            roll_seqs.append(
                (id_, items[:k])
            )

    return roll_seqs


def main():

    dataset = getattr(freerec.data.datasets.sequential, cfg.dataset)(root=cfg.root)
    User, Item = dataset.fields[USER, ID], dataset.fields[ITEM, ID]

    if cfg.neg_pool == 'all':
        # trainpipe
        trainpipe = freerec.data.postprocessing.source.RandomShuffledSource(
            source=to_roll_seqs(dataset.train(), 2)
        ).sharding_filter().seq_train_sampling_(
            dataset, leave_one_out=True, 
            num_negatives=1, pool=cfg.neg_pool # yielding (users, seqs, positives, negatives)
        ).lprune_(
            indices=[1], maxlen=cfg.maxlen,
        ).add_(
            indices=[1, 2, 3], offset=NUM_PADS
        ).batch(cfg.batch_size).column_().rpad_col_(
            indices=[1], maxlen=None, padding_value=0
        ).tensor_()
    else:
        # trainpipe
        trainpipe = freerec.data.postprocessing.source.RandomShuffledSource(
            source=to_roll_seqs(dataset.train(), 2)
        ).sharding_filter().seq_train_sampling_(
            dataset, leave_one_out=True, 
            num_negatives=cfg.num_negs, pool=cfg.neg_pool # yielding (users, seqs, positives, negatives)
        ).lprune_(
            indices=[1], maxlen=cfg.maxlen,
        ).add_(
            indices=[1, 2, 3], offset=NUM_PADS
        ).batch(cfg.batch_size).column_().rpad_col_(
            indices=[1], maxlen=None, padding_value=0
        ).tensor_()


    validpipe = freerec.data.dataloader.load_seq_rpad_validpipe(
        dataset, cfg.maxlen, 
        NUM_PADS, padding_value=0,
        batch_size=100, ranking=cfg.ranking
    )
    testpipe = freerec.data.dataloader.load_seq_rpad_testpipe(
        dataset, cfg.maxlen, 
        NUM_PADS, padding_value=0,
        batch_size=100, ranking=cfg.ranking
    )

    Item.embed(
        cfg.embedding_dim, padding_idx=0
    )
    tokenizer = FieldModuleList(dataset.fields)
    model = GRU4Rec(
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

    if cfg.loss.lower() in ('bce', 'nce', 'neg'):
        criterion = freerec.criterions.BCELoss4Logits(reduction='mean')
    elif cfg.loss.lower() == 'bpr':
        criterion = freerec.criterions.BPRLoss(reduction='mean')
    elif cfg.loss.lower() in ('ce', 'ce-sampled', 'sce', 'ce-tight'):
        criterion = freerec.criterions.CrossEntropy4Logits(reduction='mean')
    else:
        raise NotImplementedError(f"{cfg.loss} is not supported ...")

    coach = CoachForGRU4Rec(
        dataset=dataset,
        trainpipe=trainpipe,
        validpipe=validpipe,
        testpipe=testpipe,
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        lr_scheduler=None,
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