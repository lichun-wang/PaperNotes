* docker， dockerhub下载的，装好了espnet，/opt/espnet/,  我就没有再重新装了

* 处理数据需要用到kaldi的库，全部链接到utils下面：ln -s ../../../tools/kaldi/egs/wsj/s5/utils utils

* ==生成scp等文件==  ./local/data_prepare.sh  wav_path  trascript_path  ，生成交换文件： wav.scp, spk2utt,utt2spk,text 等

  `./local/data_prep.sh  ./data/data_aishell/wav/ ./data/data_aishell/transcript/# 注意文档的名字必须是transcript.txt`

* ==提取fbank特征==：   ./steps/make_fbank_pitch.sh  data/train/

* ==倒谱，均值归一化==： ./steps/compute_cmvn_stats.sh data/train/

* ==生成token==   echo '<unk> 1'  > /opt/espnet/egs/xx/xmuspeech/data/lang_1_char/train_units.txt

* python text2token.py -s 1 -n 1 /opt/espnet/egs/xx/xmuspeech/data/train/text | cut -f 2- -d" " | tr " " "\n" | sort | uniq| grep -v -e '^\s*$' | awk '{print $0 " " NR +1}'  >> /opt/espnet/egs/xx/xmuspeech/data/lang_1_char/train_units.txt

* ==生成json==  ./data2json.sh --feat /opt/espnet/egs/xx/xmuspeech/data/train/feats.scp /opt/espnet/egs/xx/xmuspeech/data/train/  /opt/espnet/egs/xx/xmuspeech/data/lang_1_char/train_units.txt >  ./data/train_data.json




