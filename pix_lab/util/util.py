import glob
import tensorflow.compat.v1 as tf

def load_graph(frozen_graph_filename):
    # We load the protobuf file from the disk and parse it to retrieve the
    # unserialized graph_def
    with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    # Then, we can use again a convenient built-in function to import a graph_def into the
    # current default Graph
    with tf.Graph().as_default() as graph:
        tf.import_graph_def(
            graph_def,
            input_map=None,
            return_elements=None,
            name="",
            op_dict=None,
            producer_op_list=None
        )
    return graph

def get_F_value(path_to_eval_file):
    g = open(path_to_eval_file)
    for p in g.readlines():
        if "Resulting F_1 value: " in p:
            aStr = p[-7:-1]
            fStr = aStr.replace(".", ",")
            return fStr
