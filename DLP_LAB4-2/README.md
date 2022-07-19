修正: 
train跟test的dataloader要不一樣。隨機裁剪、翻轉是為train data增加多樣性、增加training難度。
testing時則應降低難度，取消裁剪、翻轉，才能使test accuracy提高。