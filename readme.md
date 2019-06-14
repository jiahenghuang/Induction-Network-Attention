##Induction-Network-Attention

准备做小样本分类，开始复现
Few-Shot Text Classification with Induction Network https://arxiv.org/abs/1902.10482

StructuredSelfAttention+Relation Networkhttps://github.com/laohur/LearningToCompare_FSL 
1.StructuredSelfAttention 作为编码器，传入之前展开为batch*seq*emb batch*outdim
    5*50 re95   95*50 re5 cat->475*100  
2.Relation Network 传入特征为向量
    475*100->475
3.改了train，再改valid

4.loss不动，特征编码正常，但关系计算很均匀。
