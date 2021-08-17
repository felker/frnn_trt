import onnx_graphsurgeon as gs
import numpy as np
import sys
import onnx

graph = gs.import_onnx(onnx.load("./input.onnx"))
tensors = graph.tensors()
#nodes = graph.nodes

batch_sizes = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]

for bs in batch_sizes:
	tensors["input_2"].shape[0] = bs
	tensors["time_distributed"].shape[0] = bs
	tensors["shape_tensor1"] = gs.Constant(
		name='shape_tensor1', values=np.array([bs, 128, 1], dtype=np.int32))
	graph.nodes[-1].inputs[-1] = gs.Constant(
		name='shape_tensor1', values=np.array([bs, 128, 1], dtype=np.int32)
	onnx.save(gs.export_onnx(graph.cleanup().toposort()), f"frnn-bs{bs}-seq128.onnx")

# dynamic 
bs = -1 
tensors["input_2"].shape[0] = gs.Tensor.DYNAMIC
tensors["time_distributed"].shape[0] = gs.Tensor.DYNAMIC
tensors["shape_tensor1"] = gs.Constant(
	name='shape_tensor1', values=np.array([gs.Tensor.DYNAMIC, 128, 1], dtype=np.int32))
graph.nodes[-1].inputs[-1] = gs.Constant(
	name='shape_tensor1', values=np.array([gs.Tensor.DYNAMIC, 128, 1], dtype=np.int32)
onnx.save(gs.export_onnx(graph.cleanup().toposort()), f"frnn-bs-dynamic-seq128.onnx")

