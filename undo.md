#### dataset
cinc2017:
- interface for choosing a portion of the original data
- return multiple signals under contrastive paradigm by __getitem__
chapman:
- not implement label returning thus cannot used for supvervised learning because there are much more label which conflicts with original paper
- can only flatten the leads dimension for time contrastive learning
- downsample transformation

#### train
- learning rate decay