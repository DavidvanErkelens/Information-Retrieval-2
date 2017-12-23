from graphvec import *
import argparse
import pickle
import os


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='reuters')
parser.add_argument('--save_name', type=str, default='save_name')

parser.add_argument('--friendly_print', action='store_true')
parser.add_argument('--print_freq', type=int, default=10)
parser.add_argument('--backup_freq', type=int, default=10)

parser.add_argument('--epochs', type=int, default=1000)
parser.add_argument('--learning_rate', type=float, default=1e-2)
parser.add_argument('--dropout', type=float, default=0.0)
parser.add_argument('--pos_sample_size', type=int, default=128)
parser.add_argument('--embedding_size_w', type=int, default=128)
parser.add_argument('--embedding_size_d', type=int, default=2)
parser.add_argument('--n_neg_samples', type=int, default=64)
parser.add_argument('--window_size', type=int, default=8)
parser.add_argument('--window_batch_size', type=int, default=128)
parser.add_argument('--h_layers', nargs='+', type=int, default=[32, 8])

parser.add_argument('--load_model', type=str)
parser.add_argument('--train', action='store_true')
parser.add_argument('--eval', action='store_true')
args = parser.parse_args()

def load_dataset(dataset):
    ''' Return tokenized dataset, detokenizer and tokenizer '''
    if dataset == 'reuters':
        print('Loading REUTERS dataset')
        word2id = np.load('../data/reuters/reuters_word2id.npy').item(0)
        id2word = np.load('../data/reuters/reuters_id2word.npy').item(0)
        tokenized = np.load('../data/reuters/reuters_tokenized.npy')
    elif dataset == 'alternative':
        print('Loading ALTERNATIVE dataset')
        word2id = np.load('../data/20_newsgroup/20newsgroup_word2id.npy').item(0)
        id2word = np.load('../data/20_newsgroup/20newsgroup_id2word.npy').item(0)
        tokenized = np.load('../data/20_newsgroup/20newsgroup_tokenized.npy')
    else:
        print('Unknown dataset: ', dataset)

    return tokenized, word2id, id2word

def load_dataset(dataset):
    ''' Return tokenized dataset, detokenizer and tokenizer '''
    if dataset == 'reuters':
        print('Loading REUTERS dataset')
        word2id = np.load('../data/reuters/reuters_word2id.npy').item(0)
        id2word = np.load('../data/reuters/reuters_id2word.npy').item(0)
        tokenized = np.load('../data/reuters/reuters_tokenized.npy')
    elif dataset == 'alternative':
        print('Loading ALTERNATIVE dataset')
        word2id = np.load('../data/20_newsgroup/20newsgroup_word2id.npy').item(0)
        id2word = np.load('../data/20_newsgroup/20newsgroup_id2word.npy').item(0)
        tokenized = np.load('../data/20_newsgroup/20newsgroup_tokenized.npy')
    else:
        print('Unknown dataset: ', dataset)

    return tokenized, word2id, id2word


if __name__ == "__main__":
    # Load dataset
    tokenized, word2id, id2word = load_dataset(args.dataset)

    # Create corpus
    corpus = {'tokenized': tokenized, 'word2id': word2id}

    # Initiate model
    tf.reset_default_graph()
    geo_vec_model = GraphVec(corpus=corpus, vocab_size=len(word2id),
                             h_layers=args.h_layers, learning_rate=args.learning_rate,
                             act=tf.nn.relu, dropout=args.dropout,
                             pos_sample_size=args.pos_sample_size,
                             embedding_size_w=args.embedding_size_w,
                             embedding_size_d=args.embedding_size_d,
                             n_neg_samples=args.n_neg_samples,
                             window_size=args.window_size,
                             window_batch_size=args.window_batch_size,
                             friendly_print=args.friendly_print)

    # Load model
    if args.load_model:
        geo_vec_model.load(args.load_model)

    if args.train:
        # Start training
        print("TRAINING")
        geo_vec_model.train(args.epochs, args.print_freq, args.backup_freq, save_name=args.save_name)
    elif args.eval:
        print("EVALUATING")
        with open('../data/reuters/reuters_triplets.p', 'rb') as f:
            geo_vec_model.eval_triplets(pickle.load(f))
    else:
        os.mkdir(os.path.join('doc_emb', args.save_name))
        geo_vec_model.get_doc_embeddings(os.path.join('doc_emb', args.save_name))
