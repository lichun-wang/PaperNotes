* 在提取FBank特征的时候，默认是提取25ms一帧，shift为10ms；对应参数在kaldi中设置路径：kaldi/src/feat/feature-window.h，如果需要修改在这里面修改人，然后重新编译kaldi。
* 在bin/asr_train.py中，进行了dict的修正，加入了<mask>， <blank>,<eos>
* 接着执行 asr/pytorch_backend/asr.py, 

