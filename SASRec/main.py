

import torch, math
import torch.nn as nn

import freerec
from freerec.data.fields import FieldModuleList
from freerec.data.tags import USER, SESSION, ITEM, TIMESTAMP, ID

from sampler import *

freerec.declare(version='0.6.3')

cfg = freerec.parser.Parser()

# SASRec
cfg.add_argument("--maxlen", type=int, default=50)
cfg.add_argument("--num-heads", type=int, default=1)
cfg.add_argument("--num-blocks", type=int, default=2)
cfg.add_argument("--hidden-size", type=int, default=64)
cfg.add_argument("--dropout-rate", type=float, default=0.2)

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
cfg.add_argument("--neg-pool", choices=("unseen", "non-target", "all"), default='all', help="the pool from which the negatives are sampled")

cfg.set_defaults(
    description="SASRec",
    root="../../data",
    dataset='MovieLens1M_550_Chron',
    epochs=200,
    batch_size=128,
    optimizer='adam',
    lr=1e-3,
    weight_decay=0.,
    seed=1,
)
cfg.compile()


NUM_PADS = 1


class PointWiseFeedForward(nn.Module):

    def __init__(self, hidden_size: int, dropout_rate: int):

        super(PointWiseFeedForward, self).__init__()

        self.conv1 = nn.Conv1d(hidden_size, hidden_size, kernel_size=1)
        self.dropout1 = nn.Dropout(p=dropout_rate)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(hidden_size, hidden_size, kernel_size=1)
        self.dropout2 = nn.Dropout(p=dropout_rate)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # inputs: (B, S, D)
        outputs = self.dropout2(self.conv2(self.relu(
            self.dropout1(self.conv1(inputs.transpose(-1, -2)))
        ))) # -> (B, D, S)
        outputs = outputs.transpose(-1, -2) # -> (B, S, D)
        outputs += inputs
        return outputs


class SASRec(freerec.models.RecSysArch):

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
            "positions",
            torch.tensor(range(0, maxlen), dtype=torch.long).unsqueeze(0)
        )

        self.attnLNs = nn.ModuleList() # to be Q for self-attention
        self.attnLayers = nn.ModuleList()
        self.fwdLNs = nn.ModuleList()
        self.fwdLayers = nn.ModuleList()

        self.lastLN = nn.LayerNorm(hidden_size, eps=1e-8)

        for _ in range(num_blocks):
            self.attnLNs.append(nn.LayerNorm(
                hidden_size, eps=1e-8
            ))

            self.attnLayers.append(
                nn.MultiheadAttention(
                    embed_dim=hidden_size,
                    num_heads=num_heads,
                    dropout=dropout_rate,
                    batch_first=True # !!!
                )
            )

            self.fwdLNs.append(nn.LayerNorm(
                hidden_size, eps=1e-8
            ))

            self.fwdLayers.append(PointWiseFeedForward(
                hidden_size, dropout_rate
            ))

        # False True  True ...
        # False False True ...
        # False False False ...
        # ....
        # True indices that the corresponding position is not allowed to attend !
        self.register_buffer(
            'attnMask',
            torch.ones((maxlen, maxlen), dtype=torch.bool).triu(diagonal=1)
        )

        self.reset_parameters()

    def reset_parameters(self):
        """Initializes the module parameters."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.)
            elif isinstance(m, nn.Embedding):
                nn.init.xavier_normal_(m.weight)
            elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
                nn.init.constant_(m.weight, 1.)
                nn.init.constant_(m.bias, 0.)

    def position(self, seqs: torch.Tensor):
        positions = self.Position(self.positions) # (1, maxlen, D)
        return seqs + positions

    def after_one_block(self, seqs: torch.Tensor, padding_mask: torch.Tensor, l: int):
        # inputs: (B, S, D)
        Q = self.attnLNs[l](seqs)
        seqs = self.attnLayers[l](
            Q, seqs, seqs, 
            attn_mask=self.attnMask,
            need_weights=False
        )[0] + seqs

        seqs = self.fwdLNs[l](seqs)
        seqs = self.fwdLayers[l](seqs)

        return seqs.masked_fill(padding_mask, 0.)

    def forward(self, seqs: torch.Tensor):
        padding_mask = (seqs == 0).unsqueeze(-1)
        seqs = self.Item.look_up(seqs) # (B, S) -> (B, S, D)
        seqs *= self.Item.dimension ** 0.5
        seqs = self.embdDropout(self.position(seqs))
        seqs.masked_fill_(padding_mask, 0.)

        for l in range(self.num_blocks):
            seqs = self.after_one_block(seqs, padding_mask, l)
        
        features = self.lastLN(seqs) # (B, S, D)

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


class CoachForSASRec(freerec.launcher.SeqCoach):

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
            posLogits, negLogits, logits = self.model.predict(seqs, positives, negatives)
            indices = positives != 0
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
        trainpipe = freerec.data.postprocessing.source.RandomShuffledSource(
            source=dataset.train().to_seqs(keepid=True)
        ).sharding_filter().seq_train_sampling_(
            dataset, leave_one_out=False, 
            num_negatives=1, pool=cfg.neg_pool # yielding (user, seqs, targets, negatives)
        ).lprune_(
            indices=[1, 2, 3], maxlen=cfg.maxlen
        ).add_(
            indices=[1, 2, 3], offset=NUM_PADS
        ).lpad_(
            indices=[1, 2, 3], maxlen=cfg.maxlen, padding_value=0
        ).batch(cfg.batch_size).column_().tensor_()
    else:
        trainpipe = freerec.data.postprocessing.source.RandomShuffledSource(
            source=dataset.train().to_seqs(keepid=True)
        ).sharding_filter().seq_train_sampling_(
            dataset, leave_one_out=False, 
            num_negatives=cfg.num_negs, pool=cfg.neg_pool # yielding (user, seqs, targets, negatives)
        ).lprune_(
            indices=[1, 2, 3], maxlen=cfg.maxlen
        ).add_(
            indices=[1, 2, 3], offset=NUM_PADS
        ).lpad_(
            indices=[1, 2, 3], maxlen=cfg.maxlen, padding_value=0
        ).batch(cfg.batch_size).column_().tensor_()

    validpipe = freerec.data.dataloader.load_seq_lpad_validpipe(
        dataset, cfg.maxlen, 
        NUM_PADS, padding_value=0,
        batch_size=100, ranking=cfg.ranking
    )
    testpipe = freerec.data.dataloader.load_seq_lpad_testpipe(
        dataset, cfg.maxlen, 
        NUM_PADS, padding_value=0,
        batch_size=100, ranking=cfg.ranking
    )

    Item.embed(
        cfg.hidden_size, padding_idx = 0
    )
    tokenizer = FieldModuleList(dataset.fields)
    model = SASRec(
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
    
    if cfg.loss.lower() in ('bce', 'nce', 'neg'):
        criterion = freerec.criterions.BCELoss4Logits(reduction='mean')
    elif cfg.loss.lower() == 'bpr':
        criterion = freerec.criterions.BPRLoss(reduction='mean')
    elif cfg.loss.lower() in ('ce', 'ce-sampled', 'sce', 'ce-tight'):
        criterion = freerec.criterions.CrossEntropy4Logits(reduction='mean')
    else:
        raise NotImplementedError(f"{cfg.loss} is not supported ...")

    coach = CoachForSASRec(
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