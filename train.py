
import os, time, sys
import pickle
import torch
import torch.utils.data
from torch.utils.data import TensorDataset, ConcatDataset, DataLoader
import torch.nn.functional as F
from absl import app
from absl import flags
from ignite.metrics import Accuracy, Loss
import numpy as np
import model as MV
from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
import ssl
import gc
from ignite.handlers import ModelCheckpoint

ssl._create_default_https_context = ssl._create_unverified_context


# ------------
# parser.add_argument('--data', '-d', default='/train')
flags.DEFINE_string('--data', '/train', '')
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


# -----

def main(argv):
    START = time.time()
    del argv  # Unused.
    print(FLAGS.gpu)
    print('GPU: ', FLAGS.gpu)
    print('# Minibatch-size: ', FLAGS.batch_size)
    print('')

    #-------------------------------
    # GPU check
    xp = np
    if FLAGS.gpu >= 0:
        print('GPU mode')

    #-------------------------------
    # Loading datasets
    for i in range(5):
        #  making feature vectors of seq, one-hot encoding

        print('Making Training dataset...')
        ecfp = xp.load(FLAGS.input+'/cv_'+str(i)+'/train_fingerprint.npy')
        ecfp = xp.asarray(ecfp, dtype='float32').reshape(-1,1024)

        file_interactions=xp.load(FLAGS.input+'/cv_'+str(i)+'/train_interaction.npy')
        print('Loading labels: train_interaction.npy')
        cID = xp.load(FLAGS.input+'/cv_'+str(i)+'/train_chemIDs.npy')
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
        interactions = xp.asarray(file_interactions, dtype='int32').reshape(-1,FLAGS.n_out)
        n2vc = np.asarray(n2v_c, dtype='float32').reshape(-1, 128)
        n2vp = np.asarray(n2v_p, dtype='float32').reshape(-1, 128)
        #reset memory
        del n2v_c, n2v_p, cID, pID, modelcc, modelpp, file_interactions
        gc.collect()

        file_sequences=xp.load(FLAGS.input+'/cv_'+str(i)+'/train_reprotein.npy')
        print('Loading sequences: train_reprotein.npy', flush=True)
        sequences = xp.asarray(file_sequences, dtype='float32').reshape((-1, 1, FLAGS.pro_size, feature_vector_seq))
        # reset memory
        del file_sequences
        gc.collect()

        print('interactions.shape: ', interactions.shape, 'ecfp.shape: ', ecfp.shape,'sequences.shape: ',  sequences.shape, 'n2vc.shape:', n2vc.shape,'n2vp.shape:', n2vp.shape, flush=True)

        print('Now concatenating...', flush=True)

        dataset_ecfp = torch.utils.data.DataLoader(ecfp)
        print(type(dataset_ecfp))
        ds_con = torch.utils.data.ConcatDataset([ecfp, sequences, n2vc, n2vp, interactions])
        train_size = int(len(ds_con) * 0.8)
        val_size = int(len(ds_con)) - train_size
        train_dataset, valid_dataset = torch.utils.data.random_split(ds_con, [train_size, val_size])
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=FLAGS.batch_size, shuffle=True)
        test_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=FLAGS.batch_size, shuffle=True)

        # train_dataset, valid_dataset = train_dataset[:n], train_dataset[n:]
        # print('train: ', len(train_dataset), flush=True)
        # print('valid: ', len(valid_dataset), flush=True)

        print('pattern: ', i, flush=True)
        output_dir = FLAGS.output+'/'+'ecfpN2vc_mSGD'+'/'+'pattern'+str(i)
        os.makedirs(output_dir)

        #-------------------------------
        #reset memory again
        del sequences, interactions, ecfp, n2vc, n2vp
        gc.collect()

        # Setup an optimizer

        #-------------------------------
        # Make a specified GPU current
        device = 'cpu'
        if torch.cuda.is_available():
            device = 'cuda'
        print('use'+device)

        #-------------------------------
        # Set up a neural network to train
        print('Set up a neural network to train', flush=True)
        model = MV.DeepCNN(FLAGS.pro_size, feature_vector_seq,  FLAGS.s1, FLAGS.sa1, FLAGS.s2, FLAGS.sa2, FLAGS.s3, FLAGS.sa3, FLAGS.j1, FLAGS.pf1, FLAGS.ja1, FLAGS.j2, FLAGS.pf2, FLAGS.ja2, FLAGS.j3, FLAGS.pf3, FLAGS.ja3, FLAGS.n_hid3, FLAGS.n_hid4, FLAGS.n_hid5, FLAGS.n_out)
        if torch.cuda.is_available():
            model.cuda()
        # optimizer = chainer.optimizers.MomentumSGD(lr=0.01, momentum=0.9)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=0.00001)


        #-------------------------------
        # L2 regularization(weight decay)
        '''
        for param in model.params():
            if param.name != 'b':
                param.update_rule.add_hook(WeightDecay(0.00001))
        '''
        #-------------------------------
        # Set up a trainer
        print('Trainer is setting up...', flush=True)

        trainer = create_supervised_trainer(model, optimizer, F.nll_loss, device=device)
        evaluator = create_supervised_evaluator(model, metrics={'accuracy': Accuracy(), 'nll': Loss(F.nll_loss)}, device=device)
        training_history = {'accuracy': [], 'loss': []}
        validation_history = {'accuracy': [], 'loss': []}

        @trainer.on(Events.EPOCH_COMPLETED)
        def log_training_results(engine):
            evaluator.run(train_loader)
            metrics = evaluator.state.metrics
            avg_accuracy = metrics['accuracy']
            avg_nll = metrics['nll']
            training_history['accuracy'].append(avg_accuracy)
            training_history['loss'].append(avg_nll)
            print(
                "Training Results - Epoch: {}  Avg accuracy: {:.2f} Avg loss: {:.2f}"
                .format(engine.state.epoch, avg_accuracy, avg_nll)
            )

        @trainer.on(Events.EPOCH_COMPLETED)
        def log_validation_results(engine):
            evaluator.run(test_loader)
            metrics = evaluator.state.metrics
            avg_accuracy = metrics['accuracy']
            avg_nll = metrics['nll']
            validation_history['accuracy'].append(avg_accuracy)
            validation_history['loss'].append(avg_nll)
            print(
                "Validation Results - Epoch: {}  Avg accuracy: {:.2f} Avg loss: {:.2f}"
                .format(engine.state.epoch, avg_accuracy, avg_nll))

        checkpointer = ModelCheckpoint(
            './models',
            'model',
            save_interval=1,
            n_saved=2,
            create_dir=True,
            save_as_state_dict=True,
            require_empty=False,
        )
        trainer.add_event_handler(Events.EPOCH_COMPLETED, checkpointer, {'Model': model})

        trainer.run(train_loader, max_epochs=FLAGS.epoch)
        print('LOSS:', model(train_loader))
        END = time.time()
        print('Nice, your Learning Job is done.　Total time is {} sec．'.format(END-START))

        del model, trainer
        gc.collect()
        # my writing
        #  training part

#    logging.info('gpu id is: ', FLAGS.gpu)
#    logging.info('string test:', FLAGS.test)

'''
        # get amino acid seq from Ensembl protein ID (ENSP) and convert them to onehot vectors
        with open(FLAGS.input+'/cv_'+str(i)+FLAGS.data+'_proIDs.txt') as f:
            pID = [s.strip() for s in f.readlines()]

        plen = len(pID)
        ens = EnsemblRelease(93)  # release 93 uses human reference genome GRCh38
        to_seq = []

        for j in pID:
            seq = ens.protein_sequence(j)  # get amino acid seq from ENSP using pyensembl
            to_seq.append(seq)

        amino_acid = 'ACDEFGHIKLMNPQRSTVWY'  # define universe of possible input values
        char_to_int = dict((c, n) for n, c in enumerate(amino_acid))  # define a mapping of chars to integers
        int_to_char = dict((n, c) for n, c in enumerate(amino_acid))  # n: index, c: amino_acid
        integer_encoded = []

        for k in range(len(to_seq)):
            integer_encoded.append([char_to_int[char] for char in to_seq[k]])  # integer encode input data

        Max = 5762
        onehot_tr = np.empty((plen, Max, 20), dtype='float32')
        for l in range(len(integer_encoded)):
            b_onehot = np.identity(20, dtype='float32')[integer_encoded[l]]
            differ_tr = Max - len(integer_encoded[l])
            b_zeros = np.zeros((differ_tr, 20), dtype='float32')
            onehot_tr[l] = np.vstack((b_onehot, b_zeros))
        np.save(FLAGS.input + '/cv_' + str(i) + FLAGS.data + '_reprotein.npy', onehot_tr)
        
        print('Making training dataset')
        file_sequences = np.load(os.path.join(os.path.join(FLAGS.input, '/cv_' + str(i)), 'train_reprotein.npy'))  # in origin_DTI, xp is alias to np (numpy)
        sequences = np.asarray(file_sequences, dtype='float32').reshape(-1, 1, FLAGS.pro_size, feature_vector_seq)  # prosize in origin
        del file_sequences
        gc.collect()
'''

if __name__ == '__main__':
    app.run(main)


