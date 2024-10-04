import argparse
import utils
import MyData
import MyTrans
import os

parse = argparse.ArgumentParser(description="machine translation using transformer")
parse.add_argument("--path_train", type=str, help="path to data train", default='data/train1.json')
parse.add_argument("--path_valid", type=str, help="path to data valid", default='data/valid1.json')
parse.add_argument("--path_test", type=str, help="path to data test", default='data/test1.json')
parse.add_argument("--in_lang", type=str, help="input language", default='E')
parse.add_argument("--out_lang", type=str, help="output language", default='V')
parse.add_argument("--vocab_E", type=str, help="path Vocabulary of E", default='vocab/vocab_E1.json')
parse.add_argument("--vocab_V", type=str, help="path Vocabulary of V", default='vocab/vocab_V1.json')
parse.add_argument("--dmodel", type=int, help="Dimension of model", default=64)
parse.add_argument("--dembed", type=int, help="Dimension of embedding", default=64)
parse.add_argument("--d_ff", type=int, help="Dimension of feed-forward layer", default=128)
parse.add_argument("--head", type=int, help="Number of attention heads", default=8)
parse.add_argument("--active", type=str, help="Type of activation function", default="relu")
parse.add_argument("--layer", type=int, help="Number of layers", default=2)
parse.add_argument("--dropout", type=float, help="Dropout rate", default=0.1)
parse.add_argument("--eps", type=float, help="Epsilon value", default=1e-5)
parse.add_argument("--epoch", type= int, default= 10000, help= "epoch")
parse.add_argument("--batch_size", type= int, default= 64, help = "batch size in training and testing")
parse.add_argument("--result_path", type= str, default= None, help = "file path to result model")
parse.add_argument("--metric_path", type= str, default= None, help = "file path to result metric")
parse.add_argument("--folder", type = str, default = "", help = "folder have data, checkpoint, metric")
args = parse.parse_args()

if __name__ == "__main__":
    path_train = os.path.join(args.folder, args.path_train)
    path_test = os.path.join(args.folder, args.path_test)
    path_valid = os.path.join(args.folder, args.path_valid)
    vocab_E = os.path.join(args.folder, args.vocab_E)
    vocab_V = os.path.join(args.folder, args.vocab_V)
    result_path = os.path.join(args.folder, args.result_path)
    metric_path = os.path.join(args.folder, args.metric_path)
    data_train = MyData.EV_Data(path_train, inp = args.in_lang, out = args.out_lang, E_vocab_path= vocab_E, V_vocab_path= vocab_V)
    model = MyTrans.TransformerParallel(input_vocab_size= len(data_train.inp_vocab), output_vocab_size= len(data_train.out_vocab), dmodel = args.dmodel, dembed = args.dembed, 
                                d_ff= args.d_ff, head = args.head, active= args.active, layer= args.layer, dropout= args.dropout, eps = args.eps)     
    data_train = MyData.DataLoader(data_train, batch_size= 64, shuffle= True)
    if args.path_valid is not None:
        data_valid = MyData.EV_Data(path_valid, inp = args.in_lang, out = args.out_lang, E_vocab_path= vocab_E, V_vocab_path= vocab_V)
        data_valid = MyData.DataLoader(data_valid, batch_size= args.batch_size, shuffle= False)
        data_test = MyData.EV_Data(path_test, inp = args.in_lang, out = args.out_lang, E_vocab_path= vocab_E, V_vocab_path= vocab_V)
        data_test = MyData.DataLoader(data_test, batch_size= args.batch_size, shuffle= False)
    else:
        data_valid = None
        data_test = None
 
    optimizer = utils.optim.Adam(model.parameters(), lr = 1.0, betas= (0.9, 0.98), eps= 1e-9)     
    utils.train_parallel(model, optimizer, args.epoch, data_train, data_valid, data_test, result_path, metric_path)