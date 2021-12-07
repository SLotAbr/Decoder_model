# Decoder_model
Decoder model for language modelling

Put your text in "input.txt" and use python 3.* + numpy framework for running Main_loop.py. Also remember about making "parameters" folder.

Be careful with high learning rate: overflowing may occur in exp().

I took some code snippets of Main_loop.py from Andrej Karpathy's [RNN_Char_Level.py](https://gist.github.com/karpathy/d4dee566867f8291f086). If you still haven't seen it or the original [article](https://karpathy.github.io/2015/05/21/rnn-effectiveness/), then I highly recommend do it: the article and the code have not just become very popular.

# Some notes
This architecture doesn't work efficiently on char-level: it's unclear, how to distribute attention between letters. The model achieves much better results on word-level modelling (Byte pair encode also can improve performance).

# Some useful links
- [Paper about original Transformer model](https://arxiv.org/abs/1706.03762)
- Illustrated Jay Alammar's articles about [Transformer](https://jalammar.github.io/illustrated-transformer/) and [GPT-2 architecture](https://jalammar.github.io/illustrated-gpt2/)
- [Detailed batchnorm backpropagation calculating](https://chrisyeh96.github.io/2017/08/28/deriving-batchnorm-backprop.html) - I used this to understand how I could calculate derivatives through the Layer Norm

# Possible improvements
- more efficient MH_attention_mechanism and LayerNorm for evaluation phase
	(or STOP recalculating existing values for previous tokens!)
- correspond module for eval phase in Decoder_model class
- multiprocessing feature for Circle operations (e.g. head's calculating)
