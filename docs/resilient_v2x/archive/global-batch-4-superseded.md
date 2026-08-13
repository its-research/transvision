# Global-Batch-4 Superseded Task Archive

| 字段 | 值 |
| :--- | :--- |
| 状态 | Archived / superseded |
| 归档时间 | 2026-08-10 04:23 UTC+8 |
| 原训练条件 | 4 GPU × 每卡 batch size 1 = global batch size 4 |
| 最终训练条件 | 4 GPU × 每卡 batch size 2 = global batch size 8 |
| 处理 | 旧评测任务停止并移出队列；旧训练任务和日志保留 |
| 最终比较资格 | 不使用旧 checkpoint 或旧评测任务 |

| 方法 / 变体 | 旧训练任务 | 已停止旧评测 | global-batch-8 训练 | 新 1337×12 评测 |
| :--- | :--- | :--- | :--- | :--- |
| Resilient V2X | `77afadda645f44748e1236eb91b5e664` | `6e3aae4a75754419a6139dd06bc170bb` | `d7fd0d8488b64a3aa0f5dea4c9ea6f06` | `80066fc18bde47509c9dd936017d1f5c` |
| V2X-ViT-style | `09bafcdff19f4da992f53d2ccbfdfc12` | `8cd022cfae6d4e07911f600eaa5dbe36` | `a0d376ed58ac4ea6a696c2f43f38ba93` | `f532345fcf0c436b87373b3e8a0aaf88` |
| CoBEVT-style | `9ff8802857c843ffba8de0eea92bab6a` | `4b94c634fc964246a42a11ba43d6d097` | `ce80972f63cc47f39992558b1db8c10f` | `b9837e05c59244bd988fba33037cd57e` |
| CoFormerNet-style | `cc18359bbc314429bd90c45fad0b2cd0` | `01520e68dce34cb7b8c9490c29254cd8` | `4861fb5617ca4a7787644929d6fb116f` | `070d2536f10448369193a9a1d9c5d2ce` |
| MIT-HAN BEVFusion-style | `5e0060f09c58431fb39bbdea6c5448e8` | `7ee3f4044ff0442aac4e23775fff6d83` | `0507294403a04f609ed191b5fa02f68e` | `96c3b510a7c14444941d8cd0ef7bbbe0` |
| FFNet-style | `ddbeeec499fb4b55bcd13bc14823b9df` | `4a35cfd3908f457286b578063a84eb71` | `d14539b0ae47416d80f5ac5a260baa01` | `01d6ba9015244da98b66eacef578d5ff` |
| No PTF | `327c0a46afad439289402a4c155d9650` | `d23897a7068245d28db31462d9c4c5f2` | `494f4c7f26c244558ec95fec722add7f` | `9aaae56030b845dda35345e0bbb77fc9` |
| Linear PTF | `0d84d9269d9345adb126bf5dca43b2e6` | `7f47c8f05cb14d97a8d770117e80b9cf` | `83fd1c2a99b741cda32255dd563d5f65` | `5222fb74f7ed4553a78b4aa5723d32ea` |
| Static three-expert | `cbee478f2d084dc19d5baf15a48355a4` | `16c2ad5127b346d69af53b4edec46c0c` | `32500ba25a1a4333947f96a17261bc16` | `72976ba977134d1a82340ea508476c27` |
| Uniform gate | `dfd25f7cab354cb78532199120906a0a` | `1072c505da784329a87e2dfc5b004c0f` | `1ce0396eb1094c2c828f1943490b4df4` | `f5d43a75df154e3187285ba6943a9fd3` |
| No reliability | `aadbc8aba0f9484d89274d6967c5e0a7` | `2d1850127c804489a54b1d9deaa151cc` | `cab5235c38cb41d4ad37880d71e5d14a` | `9ade738e28554c5c80c13ccf55cea86e` |
| No delay metadata | `3d8b301acff844368f8a0a4cfb50efe4` | `9abafbdc2c8f4f8182d9279fc8a1a0c6` | `34ca778640ec44ca9e6f53e1d1be34f6` | `67351634c0284917ab665de74f0baa71` |
| No distillation | `8a8fc0f9e26640e582fecf202f71613e` | `56fc6349a8c94b64a375beacdf184dd9` | `1060483084cd428aa457cf619cbedec6` | `3f89473f2cb6434f99d8fd7f52d5d507` |
| Capacity-matched concat | `6a94c21ecd1a4948b5f751d4e63a1e38` | `b6d13ffd044a4d2ab5921804dc969f6e` | `67918c82bba84776bd5d6f7c26f54906` | `a29de3e9665c4d30b68470086ed59cae` |
