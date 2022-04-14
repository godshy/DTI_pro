import ssl
import os
import time
from absl import app
from absl import flags
import torch
import pickle
import numpy as np
import os
import torch
import gc

ssl._create_default_https_context = ssl._create_unverified_context

# parser.add_argument('--data', '-d', default='/train')
# flags.DEFINE_string('--data', '/train', '')
# ------------
flags.DEFINE_integer('gpu', 1, '-g=GPU_ID or --gpu=GPU_ID', short_name='g')  # -g=id or --gpu=id
flags.DEFINE_integer('batch_size', 100, '--batch_size=MINI_BATCH_SIZE')  # 100 batches of 1/150 epoch
flags.DEFINE_integer('epoch', 150, '--epoch= EPOCH_SIZE')
flags.DEFINE_integer('s1', 1, '')
flags.DEFINE_integer('sa1', 1, '')
flags.DEFINE_integer('s2', 1, '')
flags.DEFINE_integer('sa2', 1, '')
flags.DEFINE_integer('s3', 1, '')
flags.DEFINE_integer('sa3', 1, '')
flags.DEFINE_integer('j1', 33, '')
flags.DEFINE_integer('ja1', 17, '')
flags.DEFINE_integer('pf1', 64, '')
flags.DEFINE_integer('j2', 23, '')
flags.DEFINE_integer('pf2', 64, '')
flags.DEFINE_integer('ja2', 11, '')
flags.DEFINE_integer('j3', 33, '')
flags.DEFINE_integer('pf3', 32, '')
flags.DEFINE_integer('ja3', 17, '')
flags.DEFINE_integer('n_hid3', 70, '')
flags.DEFINE_integer('n_hid4', 80, '')
flags.DEFINE_integer('n_hid5', 60, '')
flags.DEFINE_integer('n_out', 1, '')
flags.DEFINE_integer('pro_size', 5762, '')
# flags.Define_
flags.DEFINE_integer('frequency', 1, '')
# set input dir (need to be revised)
flags.DEFINE_string('input', './dataset_hard', '--input=./path/to/your/input')
# set output dir
flags.DEFINE_string('output', './result/dataset_hard', '--output=./path/to/your/output')
# os.path.join("/A/B/C", "file.py")

FLAGS = flags.FLAGS

# Required flag.
# flags.mark_flag_as_required("XXX")

# -----
feature_vector_seq = 20  # size of feature vector
def main(argv):
    print(FLAGS.gpu)
    print('GPU: ', FLAGS.gpu)
    print('Minibatch-size: ', FLAGS.batch_size)
    if FLAGS.gpu >= 0:
        print('GPU mode')
        print('Is cuda available:  ', torch.cuda.is_available())

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('use'+device)

    for i in range(5):
        #  making feature vectors of seq, one-hot encoding

        print('Making Training dataset...')
        ecfp = np.load(FLAGS.input+'/cv_'+str(i)+'/train_fingerprint.npy')
        ecfp = np.asarray(ecfp, dtype='float32').reshape(-1, 1024)

        file_interactions=np.load(FLAGS.input+'/cv_'+str(i)+'/train_interaction.npy')
        print('Loading labels: train_interaction.npy')
        cID = np.load(FLAGS.input+'/cv_'+str(i)+'/train_chemIDs.npy')
        print('Loading chemIDs: train_chemIDs.npy')
        with open(FLAGS.input+'/cv_'+str(i)+'/train_proIDs.txt') as f:
            pID = [s.strip() for s in f.readlines()]
        print('Loading proIDs: train_proIDs.txt')
        n2v_c, n2v_p = [], []
        with open('./modelpp.pickle', mode='rb') as f:
            modelpp = pickle.load(f)
        with open('./modelcc.pickle', mode='rb') as f:
            modelcc = pickle.load(f)
        for j in cID:
            n2v_c.append(modelcc.wv[str(j)])
        for k in pID:
            n2v_p.append(modelpp.wv[k])
        interactions = np.asarray(file_interactions, dtype='int32').reshape(-1,FLAGS.n_out)
        n2vc = np.asarray(n2v_c, dtype='float32').reshape(-1, 128)
        n2vp = np.asarray(n2v_p, dtype='float32').reshape(-1, 128)
        #reset memory
        del n2v_c, n2v_p, cID, pID, modelcc, modelpp, file_interactions
        gc.collect()

        file_sequences = np.load(FLAGS.input+'/cv_'+str(i)+'/train_reprotein.npy')
        print('Loading sequences: train_reprotein.npy', flush=True)
        sequences = np.asarray(file_sequences, dtype='float32').reshape((-1, 1, FLAGS.pro_size, feature_vector_seq))
        # reset memory
        del file_sequences
        gc.collect()
        print('pattern_', i)
        print('interactions.shape: ', interactions.shape, 'ecfp.shape: ', ecfp.shape, 'sequences.shape: ',  sequences.shape, 'n2vc.shape:', n2vc.shape, 'n2vp.shape:', n2vp.shape, flush=True)


if __name__ == '__main__':
    app.run(main)