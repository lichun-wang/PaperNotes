### 一、yaml的使用

```python
import yaml

with open("./config.yml", 'r') as f:
configs = yaml.load(f)
```

二、logging的使用

```python
import logging
from logging import handlers

class Logger(object):
    level_relations = {
        'debug':logging.DEBUG,
        'info':logging.INFO,
        'warning':logging.WARNING,
        'error':logging.ERROR,
        'crit':logging.CRITICAL
    }

    def __init__(self,filename,level='info',when='D',backCount=3,fmt='%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s'):
        self.logger = logging.getLogger(filename)
        format_str = logging.Formatter(fmt)
        self.logger.setLevel(self.level_relations.get(level))
        sh = logging.StreamHandler()
        sh.setFormatter(format_str)
        th = handlers.TimedRotatingFileHandler(filename=filename,when=when,backupCount=backCount,encoding='utf-8')
        th.setFormatter(format_str)
        self.logger.addHandler(sh)
        self.logger.addHandler(th)

 ## 设置路径
log = Logger(filename='./log/log.txt', level='debug')

## 写入

log.logger.info(configs)

log.logger.info (('Epoch: [{0}][{1}/{2}], lr: {lr:.6f}\t'
'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
'Loss {loss.val:.3f} ({loss.avg:.4f})\t'
'Acc {acc.val:.3f} ({acc.avg:.4f})\t'.format(
epoch, i, len(train_loader), batch_time=batch_time,
data_time=data_time, loss=losses, lr=optimizer.param_groups[-1]['lr'], acc=acces)))
```

三、tensorboardX的使用

- 参考git链接： https://github.com/lanpa/tensorboardX

```python
from tensorboardX import SummaryWriter
writer = SummaryWriter(log_dir='./log/')
writer.add_scalar("train/loss", loss, epoch*len(train_loader) + i)
writer.add_scalar("train/lr", optimizer.param_groups[-1]['lr'], epoch*len(train_loader) + i)
writer.add_scalar("train/acc", acc, epoch * len(train_loader) + i)
```





