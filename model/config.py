
class Config:
    def __init__(self):
        self.learning_rate=0.0003
        self.num_classes = 2
        self.batch_size = 64
        self.sequence_length = 100
        self.vocab_size = 50000

        self.d_model =512
        self.num_layer=6
        self.h=8
        self.d_k=64
        self.d_v=64

        self.clip_gradients = 5.0
        self.decay_steps = 1000
        self.decay_rate = 0.9
        self.dropout_keep_prob = 0.9
        self.ckpt_dir = 'checkpoint/dummy_test/'
        self.is_training=True
        self.is_pretrain=True
        self.num_classes_lm=self.vocab_size
