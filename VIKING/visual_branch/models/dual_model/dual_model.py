import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb

# modules for VQA
from vqa.models import seq2vec
from vqa.models import fusion
from models import vec2seq

# Attention Module
import models.attention_modules as attention_modules


class Dual_Model(nn.Module):
    def __init__(self, opt={}, vocab_words=[], vocab_answers=[]):
        super(Dual_Model, self).__init__()
        self.opt = opt

        # To fuse different glimpses
        dim_h = int(
            self.opt["vqa"]["fusion"]["dim_hv"]
            / opt["attention"]["nb_glimpses"]
        )

        self.vocab_answers = vocab_answers
        self.vocab_words = vocab_words
        self.num_classes = len(self.vocab_answers) + 1  # for [UNK]

        # VQA Modules
        self.seq2vec = seq2vec.factory(self.vocab_words, self.opt["seq2vec"])

        # Modules for classification
        self.linear_q_fusion = nn.Linear(
            self.opt["dim_q"], self.opt["vqa"]["fusion"]["dim_hq"]
        )
        self.linear_classif = nn.Linear(
            self.opt["vqa"]["fusion"]["dim_mm"], self.num_classes
        )
        self.fusion_classif_vqa = fusion.MutanFusion(
            self.opt["vqa"]["fusion"],
            visual_embedding=False,
            question_embedding=False,
        )
        self.attention_vqa = getattr(
            attention_modules, opt["attention"]["arch"]
        )(opt, use_linear=False)
        self.linear_q_att = nn.Linear(
            self.opt["dim_q"], self.opt["attention"]["dim_q"]
        )
        self.list_linear_v_fusion_vqa = nn.ModuleList(
            [
                nn.Linear(self.opt["dim_v"], dim_h)
                for i in range(self.opt["attention"]["nb_glimpses"])
            ]
        )

        # share W and E
        self.answer_embeddings = nn.Embedding(
            self.linear_classif.out_features, self.linear_classif.in_features
        )
        # VQG modules
        self.linear_va_transform = nn.Linear(
            self.linear_classif.in_features,
            self.opt["vqg"]["vec2seq"]["dim_embedding"],
        )
        self.linear_a_att = nn.Linear(
            self.opt["dim_a"], self.opt["attention"]["dim_q"]
        )

        self.linear_a_fusion = nn.Linear(
            self.opt["vqa"]["fusion"]["dim_mm"],
            self.opt["vqa"]["fusion"]["dim_hq"],
        )
        # Modules for Question Generation
        self.question_generation = getattr(
            vec2seq, opt["vqg"]["vec2seq"]["arch"]
        )(vocab_words, opt["vqg"]["vec2seq"])

        # Sharable modules
        if self.opt.get("share_modules", True):
            print("Sharing Modules: [Attention] and [Fusion]")
            self.fusion_classif_vqg = self.fusion_classif_vqa
            self.attention_vqg = self.attention_vqa
            self.list_linear_v_fusion_vqg = self.list_linear_v_fusion_vqa
        else:
            print("Disable Module Sharing")
            self.fusion_classif_vqg = fusion.MutanFusion(
                self.opt["vqa"]["fusion"],
                visual_embedding=False,
                question_embedding=False,
            )
            self.attention_vqg = getattr(
                attention_modules, opt["attention"]["arch"]
            )(opt, use_linear=False)
            self.list_linear_v_fusion_vqg = nn.ModuleList(
                [
                    nn.Linear(self.opt["dim_v"], dim_h)
                    for i in range(self.opt["attention"]["nb_glimpses"])
                ]
            )

        self.is_testing = False
        self.sample_num = 5

    def set_share_parameters(self):
        print("Sharing [Answer Embeddings]")
        self.answer_embeddings.weight = self.linear_classif.weight
        print("Sharing parameters between [seq2vec] and [vec2seq]")
        self.question_generation.embedder = self.seq2vec.embedding
        if self.opt["seq2vec"]["arch"] == "lstm":
            print("Sharing parameters between [LSTM]")
            self.question_generation.rnn = self.seq2vec.rnn

    def forward(self, input_v, input_q=None, target_a=None):
        if self.is_testing:  # Start testing Mode
            return self._testing(input_v, input_q, target_a)
        else:
            return self._train(input_v, input_q, target_a)

    def _fusion_glimpses(self, list_v_att, fusion_list):
        # Process visual for each glimpses
        list_v = []
        for glimpse_id, x_v_att in enumerate(list_v_att):
            x_v = F.dropout(
                x_v_att,
                p=self.opt["vqa"]["fusion"]["dropout_v"],
                training=self.training,
            )
            x_v = fusion_list[glimpse_id](x_v)
            if "activation_v" in self.opt["vqa"]["fusion"]:
                x_v = getattr(F, self.opt["vqa"]["fusion"]["activation_v"])(x_v)
            list_v.append(x_v)
        x_v = torch.cat(list_v, 1)

        return x_v

    def _vqa_forward(self, input_v, input_q):
        x_q = self.seq2vec(input_q)
        # attention
        x_q_att = F.dropout(
            x_q, p=self.opt["attention"]["dropout_q"], training=self.training
        )
        x_q_att = self.linear_q_att(x_q_att)
        if "activation_q" in self.opt["attention"]:
            x_q_att = getattr(F, self.opt["attention"]["activation_q"])(x_q_att)

        att_v_list_vqa = self.attention_vqa(input_v, x_q_att)
        x_v_vqa = self._fusion_glimpses(
            att_v_list_vqa, self.list_linear_v_fusion_vqa
        )
        x_q = F.dropout(
            x_q,
            p=self.opt["vqa"]["fusion"]["dropout_q"],
            training=self.training,
        )
        x_q = self.linear_q_fusion(x_q)
        x_q_transformed = getattr(F, self.opt["vqa"]["fusion"]["activation_q"])(
            x_q
        )
        # Second multimodal fusion
        x_a_pred = self.fusion_classif_vqa(x_v_vqa, x_q_transformed)
        x_a = getattr(F, self.opt["vqa"]["classif"]["activation"])(x_a_pred)
        x_a = F.dropout(
            x_a, p=self.opt["vqa"]["classif"]["dropout"], training=self.training
        )
        answers = self.linear_classif(x_a)
        return answers, x_q, x_a_pred

    def _vqg_forward(self, input_v, target_a, input_q=None):
        x_a_embedding = self.answer_embeddings(target_a.view(1, -1)).squeeze(0)
        x_a_att = F.dropout(
            x_a_embedding,
            p=self.opt["attention"]["dropout_q"],
            training=self.training,
        )
        x_a_att = self.linear_a_att(x_a_att)
        if "activation_q" in self.opt["attention"]:
            x_a_att = getattr(F, self.opt["attention"]["activation_q"])(x_a_att)
        # attention
        att_v_list_vqg = self.attention_vqg(input_v, x_a_att)
        x_v_vqg = self._fusion_glimpses(
            att_v_list_vqg, self.list_linear_v_fusion_vqg
        )

        x_a_embedding = self.linear_a_fusion(x_a_embedding)
        x_a = getattr(F, self.opt["vqg"]["activation"])(x_a_embedding)
        # Second multimodal fusion
        x_q_pred = self.fusion_classif_vqg(x_v_vqg, x_a)
        q_embedding = getattr(F, self.opt["vqg"]["activation"])(x_q_pred)
        q_embedding = F.dropout(
            q_embedding, p=self.opt["vqg"]["dropout"], training=self.training
        )
        questions = self._generate_qestion(q_embedding, input_q)
        return questions, x_q_pred, x_a_embedding

    def _train(self, input_v, input_q, target_a):

        ##### VQA forward #######
        answers, x_q, x_a_pred = self._vqa_forward(input_v, input_q)

        ##### VQG forward #######
        questions, x_q_pred, x_a = self._vqg_forward(input_v, target_a, input_q)

        return (
            answers,
            questions,
            F.smooth_l1_loss(x_a_pred, x_a) + F.smooth_l1_loss(x_q_pred, x_q),
        )

    def _testing(self, input_v, input_q, input_a=None):

        ##### VQA forward #######
        answers = self._vqa_forward(input_v, input_q)[0]
        g_answers_score, g_answers = torch.max(F.softmax(answers, dim=1), dim=1)
        if input_a is not None:
            generated_q = self._vqg_forward(input_v, input_a)[0]
            g_answers = input_a
        else:
            generated_q = self._vqg_forward(input_v, g_answers)[0]
            # generated_q = []
            # for j in range(self.sample_num):
            #     generated_q.append(self._vqg_forward(input_v, g_answers[:, j].contiguous())[0])
        return answers, g_answers, g_answers_score, generated_q

    def set_testing(self, is_testing=True, sample_num=5):
        self.is_testing = is_testing
        self.sample_num = sample_num

    def _generate_qestion(self, x_va, input_q=None):

        # transform the fused feature to word_embedding domain (length = opts['vec2seq']['dim_embedding'])
        x_va = F.dropout(
            x_va,
            p=self.opt["vqg"]["vec2seq"]["dropout"],
            training=self.training,
        )
        x_va = getattr(F, self.opt["vqg"]["vec2seq"]["activation"])(
            self.linear_va_transform(x_va)
        )
        x_va = F.dropout(
            x_va,
            p=self.opt["vqg"]["vec2seq"]["dropout"],
            training=self.training,
        )
        if self.is_testing:
            if x_va.size(0) == 1:
                x = self.question_generation.beam_search(x_va, beam_size=3)
            else:
                x = self.question_generation.generate(x_va)

            return x
        else:
            x = self.question_generation(x_va, input_q)
            return x
