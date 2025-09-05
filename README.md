# ResearchPaper_MotionPro
CVPR2025で発表されたMotionProという論文の実装に関するレポです。
論文ではGRADIOでしたが、GOOGLE　COLABで実装するためのコードおよびローカルで学習やベンチマーク実行などの実装をするための
忘備録としてここにコードを上げてみます。
https://arxiv.org/abs/2505.20287

# COLABコード
2025年09月3日時点でのCOLABでは機能することは確認できてますが、今後、GOOGLEのアップデートにともない機能しなくなるかもしれません。
長い間安定稼働が必要であれば、ローカルやどこかのサーバーで固定CUDA,PYTHONのVERSIONで実行環境構築することをお勧めします。

# 環境設定
https://zenn.dev/takeofuture/articles/8faa9687df3ce8
にまとめてあります。またCOLABで1_環境構築.ipynb　を動かすことでCOLAB上で環境構築できます。
（G-DRIVEにコードや環境を一度だけ保存する方法を想定してます。）
