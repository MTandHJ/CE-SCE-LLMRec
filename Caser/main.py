

import torch, math
import torch.nn as nn
import torch.nn.functional as F

import freerec
from freerec.data.fields import FieldModuleList
from freerec.data.tags import USER, SESSION, ITEM, TIMESTAMP, ID

from sampler import *

freerec.declare(version='0.6.3')

cfg = freerec.parser.Parser()

# Caser
cfg.add_argument("--maxlen", type=int, default=5)
cfg.add_argument("--hidden-size", type=int, default=64)
cfg.add_argument("--dropout-rate", type=float, default=0.5)
cfg.add_argument("--num-vert", type=int, default=4, help="number of vertical filters")
cfg.add_argument("--num-horiz", type=int, default=16, help="number of horizontal filters")

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
    description="Caser",
    root="../../data",
    dataset='MovieLens1M_550_Chron',
    epochs=50,
    batch_size=512,
    optimizer='adam',
    lr=1e-3,
    weight_decay=1.e-6,
    seed=1,
)
cfg.compile()


NUM_PADS = 1


class Caser(freerec.models.RecSysArch):

    def __init__(self, fields: FieldModuleList) -> None:
        super().__init__()

        self.fields = fields
        self.User = self.fields[USER, ID]
        self.Item = self.fields[ITEM, ID]

        self.vert = nn.Conv2d(
            in_channels=1, out_channels=cfg.num_vert,
            kernel_size=(cfg.maxlen, 1), stride=1
        )
        self.horizs = nn.ModuleList([
            nn.Conv2d(
                in_channels=1, out_channels=cfg.num_horiz,
                kernel_size=(k, cfg.hidden_size)
            )
            for k in range(1, cfg.maxlen + 1)
        ])
        self.pooling = nn.AdaptiveMaxPool1d((1,))

        self.fc_in_dims = cfg.num_vert * cfg.hidden_size + cfg.num_horiz * cfg.maxlen

        self.fc1 = nn.Linear(self.fc_in_dims, cfg.hidden_size)

        self.dropout = nn.Dropout(cfg.dropout_rate)

        self.W2 = nn.Embedding(self.Item.count + NUM_PADS, cfg.hidden_size * 2, padding_idx=0)
        self.b2 = nn.Embedding(self.Item.count + NUM_PADS, 1, padding_idx=0)

        self.reset_parameters()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.)
            elif isinstance(m, nn.Embedding):
                nn.init.xavier_normal_(m.weight)
                nn.init.normal_(m.weight, std=1. / cfg.hidden_size)
        self.b2.weight.data.zero_()

    def forward(
        self, 
        seqs: torch.Tensor,
        users: torch.Tensor,
    ):

        seqEmbs = self.Item.look_up(seqs).unsqueeze(1) # (B, 1, S, D)
        userEmbs = self.User.look_up(users).squeeze(1) # (B, D)

        vert_features = self.vert(seqEmbs).flatten(1)
        horiz_features = [
            self.pooling(F.relu(conv(seqEmbs).squeeze(3))).squeeze(2)
            for conv in self.horizs
        ]
        horiz_features = torch.cat(horiz_features, dim=1)

        features = self.dropout(torch.cat((vert_features, horiz_features), dim=1))
        features = F.relu(self.fc1(features))
        features = torch.cat([features, userEmbs], dim=1) # (B, 2D)

        return features

    def matmul_with_bias(
        self,
        x: torch.Tensor, y: torch.Tensor, b: torch.Tensor
    ):
        return x.matmul(y) + b

    def predict(
        self, 
        seqs: torch.Tensor,
        users: torch.Tensor,
        positives: torch.Tensor,
        negatives: torch.Tensor
    ):
        features = self.forward(seqs, users).unsqueeze(2) # (B, 2D, 1)

        posEmbds = self.W2(positives) # (B, 1, 2D)
        posBiass = self.b2(positives) # (B, 1, 1)

        posLogits = self.matmul_with_bias(
            posEmbds, features, posBiass
        ).squeeze(-1)

        negLogits = None
        logits = None
        # Sometimes CE is faster because:
        # https://github.com/pytorch/pytorch/issues/113934
        if cfg.loss.lower() in ('ce', 'ce-tight'):
            items = self.W2.weight[NUM_PADS:].unsqueeze(0) # (1, N, 2D)
            biass = self.b2.weight[NUM_PADS:].unsqueeze(0) # (1, N, 1)
            logits = self.matmul_with_bias(
                items, features, biass
            ).squeeze(-1) # (B, N)
        else:
            negEmbds = self.W2(negatives) # (B, K, 2D)
            negBiass = self.b2(negatives) # (B, K, 1)
            negLogits = self.matmul_with_bias(
                negEmbds, features, negBiass
            ).squeeze(-1) # (B, K)
        return posLogits, negLogits, logits

    def recommend_from_pool(
        self, users: torch.Tensor, seqs: torch.Tensor, pool: torch.Tensor
    ):
        features = self.forward(seqs, users)
        itemEmbs = self.W2(pool) # (B, K, 2D)
        itemBias = self.b2(pool) # (B, K, 1)
        return self.matmul_with_bias(itemEmbs, features.unsqueeze(2), itemBias).squeeze(-1)

    def recommend_from_full(
        self, users: torch.Tensor, seqs: torch.Tensor
    ):
        features = self.forward(seqs, users)
        items = self.W2.weight[NUM_PADS:].unsqueeze(0) # (1, N, 2D)
        biass = self.b2.weight[NUM_PADS:].unsqueeze(0) # (1, N, 1)
        return self.matmul_with_bias(items, features.unsqueeze(2), biass).squeeze(-1)


class CoachForCaser(freerec.launcher.SeqCoach):

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
            posLogits, negLogits, logits = self.model.predict(seqs, users, positives, negatives.squeeze(1))
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
        if len(items) < minlen:
            roll_seqs.append(
                (id_, items)
            )
        else:
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
            source=to_roll_seqs(dataset.train(), cfg.maxlen)
        ).sharding_filter().seq_train_sampling_(
            dataset, leave_one_out=True, 
            num_negatives=1, pool=cfg.neg_pool # yielding (user, seqs, targets, negatives)
        ).lprune_(
            indices=[1], maxlen=cfg.maxlen
        ).add_(
            indices=[1, 2, 3], offset=NUM_PADS
        ).lpad_(
            indices=[1], maxlen=cfg.maxlen, padding_value=0
        ).batch(cfg.batch_size).column_().tensor_()
    else:
        # trainpipe
        trainpipe = freerec.data.postprocessing.source.RandomShuffledSource(
            source=to_roll_seqs(dataset.train(), cfg.maxlen)
        ).sharding_filter().seq_train_sampling_(
            dataset, leave_one_out=True, 
            num_negatives=cfg.num_negs, pool=cfg.neg_pool # yielding (user, seqs, targets, negatives)
        ).lprune_(
            indices=[1], maxlen=cfg.maxlen
        ).add_(
            indices=[1, 2, 3], offset=NUM_PADS
        ).lpad_(
            indices=[1], maxlen=cfg.maxlen, padding_value=0
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
        cfg.hidden_size, padding_idx=0
    )
    User.embed(cfg.hidden_size)
    tokenizer = FieldModuleList(dataset.fields)
    model = Caser(
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

    coach = CoachForCaser(
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
        cfg, monitors=[
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