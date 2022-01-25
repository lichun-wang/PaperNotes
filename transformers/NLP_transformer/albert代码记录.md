AlbertLayer :

 return  hidden_states   +  attention_output[1:]  

==hidden states是ffn的输出， atten[1:]代表的是attention probs , 另外atten[0]   是value * atten prob的结果。==



albert的输出也是包含两个， 

* output[0] 是sequence_state,  shape是[batch,  len,  num_hidden]; 
* output[1]是pooler的output，cls_state,  shape 是[batch, num_hidden]; 主要用于句子分类。



train:

* loss包含两部分，一个是MLM， 一个是SOP；MLM利用的是sequence_ouput的输出， SOP利用的是pooler_ouput的输出



网络结构：

* 需要注意的就是在attention中的两个residual的操作，如下图所示。



```python
AlbertModel(
  (embeddings): AlbertEmbeddings(
    (word_embeddings): Embedding(30000, 128, padding_idx=0)
    (position_embeddings): Embedding(512, 128)
    (token_type_embeddings): Embedding(2, 128)
    (LayerNorm): LayerNorm((128,), eps=1e-12, elementwise_affine=True)
    (dropout): Dropout(p=0.1, inplace=False)
  )
  (encoder): AlbertTransformer(
    (embedding_hidden_mapping_in): Linear(in_features=128, out_features=768, bias=True)
    (albert_layer_groups): ModuleList(
      (0): AlbertLayerGroup(
        (albert_layers): ModuleList(
          (0): AlbertLayer(
            (full_layer_layer_norm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
            (attention): AlbertAttention(
              (query): Linear(in_features=768, out_features=768, bias=True)
              (key): Linear(in_features=768, out_features=768, bias=True)
              (value): Linear(in_features=768, out_features=768, bias=True)
              (attention_dropout): Dropout(p=0.1, inplace=False)
              (output_dropout): Dropout(p=0.1, inplace=False)
              (dense): Linear(in_features=768, out_features=768, bias=True)
              (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
            )
            (ffn): Linear( biin_features=768, out_features=3072,as=True)
            (ffn_output): Linear(in_features=3072, out_features=768, bias=True)
            (dropout): Dropout(p=0.1, inplace=False)
          )
        )
      )
    )
  )
  (pooler): Linear(in_features=768, out_features=768, bias=True)
  (pooler_activation): Tanh()
)
```

![image-20210906171600477](C:\Users\wanglichun\Desktop\TyporaPapers\images\image-20210906171600477.png)



bert在训练中为了并行化，是将变长的数据，==通过pad到相同长度==，进行的训练。

bert中的word_embedding 的 size是  vocab_size * embedding_size

embedding 是一个 lookup table过程：A simple lookup table that stores embeddings of a fixed dictionary and size.
