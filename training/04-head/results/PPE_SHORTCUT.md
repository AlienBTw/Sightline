# PPE shortcut eval

Clip score: sigmoid of the factor logit pooled over every patch of every window.
Window score: max sigmoid(patch evidence) inside the window.
Pexels held-out is test + demo_eval. Voxel51 scores are val and test.
F2 and F5 labels are the Muse Spark clip verdicts. A null verdict is excluded.
A split with only one class is n/a.

Domain AUROC separates Voxel51 (positive class) from Pexels among held-out clips
whose factor label is present. Above 0.5 means the Voxel51 clips score higher
than Pexels clips with the same present label. The threshold is best F1 on
Pexels validation. pos above and neg above are the fractions of Pexels held-out
positives and negatives at or above that cut. The interval is a 1000-draw
bootstrap of held-out AUROC, seed 0, 2.5 and 97.5 percentiles.
Resamples that contain only one class are dropped.

Pexels clips: demo_eval 13, test 5, train 50, val 19.
Voxel51 clips: test 117, val 93.
PPE labels replaced F2/F5 on 658 Voxel51 clips; 0 clips had no PPE row and stay unlabeled; 210 val/test clips carry a PPE row.

## Summary

| checkpoint | factor | held-out pos/neg | AUROC | AP | AUROC 95% CI | domain AUROC | pos above | neg above | window AUROC |
|---|---|---|---|---|---|---|---|---|---|---|
| h3_wide | F2 | 12/6 | 0.403 | 0.640 | [0.117, 0.736] | 0.972 | 0.417 | 0.500 | 0.494 |
| h3_wide | F5 | 16/2 | 0.875 | 0.986 | [0.688, 1.000] | 0.945 | 0.875 | 0.000 | 0.870 |
| ppe_ft_all | F2 | 12/6 | 0.556 | 0.788 | [0.277, 0.827] | 1.000 | 0.917 | 1.000 | 0.712 |
| ppe_ft_all | F5 | 16/2 | 0.750 | 0.967 | [0.471, 1.000] | 1.000 | 0.375 | 0.000 | 0.749 |
| ppe_ft_frz | F2 | 12/6 | 0.389 | 0.663 | [0.125, 0.728] | 1.000 | 0.667 | 0.833 | 0.196 |
| ppe_ft_frz | F5 | 16/2 | 0.750 | 0.969 | [0.511, 0.941] | 1.000 | 0.125 | 0.000 | 0.736 |
| ppe_ft_frz_w05 | F2 | 12/6 | 0.194 | 0.530 | [0.000, 0.444] | 1.000 | 0.750 | 1.000 | 0.209 |
| ppe_ft_frz_w05 | F5 | 16/2 | 0.875 | 0.986 | [0.688, 1.000] | 1.000 | 0.375 | 0.000 | 0.775 |
| ppe_pex_all | F2 | 12/6 | 0.583 | 0.817 | [0.292, 0.850] | 0.813 | 0.917 | 1.000 | 0.708 |
| ppe_pex_all | F5 | 16/2 | 0.812 | 0.977 | [0.600, 1.000] | 0.946 | 0.375 | 0.000 | 0.822 |
| ppe_pex_frz | F2 | 12/6 | 0.194 | 0.525 | [0.000, 0.501] | 0.538 | 0.333 | 0.833 | 0.178 |
| ppe_pex_frz | F5 | 16/2 | 0.844 | 0.982 | [0.647, 1.000] | 1.000 | 0.250 | 0.000 | 0.808 |
| ppe_sam_frz | F2 | 12/6 | 0.625 | 0.764 | [0.333, 0.911] | 0.914 | 0.833 | 0.667 | 0.539 |
| ppe_sam_frz | F5 | 16/2 | 0.250 | 0.841 | [0.000, 0.600] | 0.862 | 0.812 | 1.000 | 0.276 |
| ppe_sam_shwd_all | F2 | 12/6 | 0.792 | 0.899 | [0.549, 1.000] | 0.671 | 0.917 | 0.500 | 0.648 |
| ppe_sam_shwd_all | F5 | 16/2 | 0.656 | 0.951 | [0.353, 0.941] | 0.843 | 0.562 | 0.500 | 0.542 |
| ppe_sam_shwd_frz | F2 | 12/6 | 0.486 | 0.760 | [0.219, 0.750] | 0.815 | 1.000 | 1.000 | 0.469 |
| ppe_sam_shwd_frz | F5 | 16/2 | 0.375 | 0.888 | [0.062, 0.706] | 0.701 | 0.562 | 1.000 | 0.290 |

## h3_wide

Checkpoint `/mnt/c/projects/Meta Global AI Developer Hackathon/Sightline/training/04-head/checkpoints/h3_wide.pt`, hidden 512, dropout 0.2.

### Pexels clip metrics

| factor | split | pos | neg | AUROC | AP | AUROC 95% CI |
|---|---|---|---|---|---|---|
| F2 | test | 4 | 1 | 0.750 | 0.950 |  |
| F2 | demo_eval | 8 | 5 | 0.300 | 0.544 |  |
| F2 | held_out | 12 | 6 | 0.403 | 0.640 | [0.117, 0.736] |
| F2 | val | 8 | 9 | 0.681 | 0.621 |  |
| F2 | train | 29 | 17 | 0.481 | 0.655 |  |
| F5 | test | 5 | 0 | n/a | n/a |  |
| F5 | demo_eval | 11 | 2 | 0.909 | 0.986 |  |
| F5 | held_out | 16 | 2 | 0.875 | 0.986 | [0.688, 1.000] |
| F5 | val | 10 | 9 | 0.578 | 0.705 |  |
| F5 | train | 31 | 18 | 0.636 | 0.731 |  |

### Held-out score distributions

| factor | group | n | mean | median |
|---|---|---|---|---|
| F2 | Pexels present (test+demo_eval) | 12 | 0.858 | 0.855 |
| F2 | Pexels absent (test+demo_eval) | 6 | 0.877 | 0.875 |
| F2 | Voxel51 present (val+test) | 142 | 0.977 | 0.993 |
| F5 | Pexels present (test+demo_eval) | 16 | 0.718 | 0.748 |
| F5 | Pexels absent (test+demo_eval) | 2 | 0.550 | 0.550 |
| F5 | Voxel51 present (val+test) | 146 | 0.950 | 0.985 |

### Shortcut

| factor | domain AUROC | n Voxel51 | n Pexels | val threshold | pos above | neg above | window AUROC | window pos/neg |
|---|---|---|---|---|---|---|---|---|
| F2 | 0.972 | 142 | 12 | 0.868 | 0.417 | 0.500 | 0.494 | 58/39 |
| F5 | 0.945 | 146 | 16 | 0.575 | 0.875 | 0.000 | 0.870 | 81/16 |

## ppe_ft_all

Checkpoint `/mnt/c/projects/Meta Global AI Developer Hackathon/Sightline/training/04-head/checkpoints/ppe_ft_all.pt`, hidden 512, dropout 0.2.

### Pexels clip metrics

| factor | split | pos | neg | AUROC | AP | AUROC 95% CI |
|---|---|---|---|---|---|---|
| F2 | test | 4 | 1 | 0.500 | 0.887 |  |
| F2 | demo_eval | 8 | 5 | 0.600 | 0.768 |  |
| F2 | held_out | 12 | 6 | 0.556 | 0.788 | [0.277, 0.827] |
| F2 | val | 8 | 9 | 0.583 | 0.550 |  |
| F2 | train | 29 | 17 | 1.000 | 1.000 |  |
| F5 | test | 5 | 0 | n/a | n/a |  |
| F5 | demo_eval | 11 | 2 | 0.636 | 0.927 |  |
| F5 | held_out | 16 | 2 | 0.750 | 0.967 | [0.471, 1.000] |
| F5 | val | 10 | 9 | 0.811 | 0.890 |  |
| F5 | train | 31 | 18 | 1.000 | 1.000 |  |

### Held-out score distributions

| factor | group | n | mean | median |
|---|---|---|---|---|
| F2 | Pexels present (test+demo_eval) | 12 | 0.752 | 0.828 |
| F2 | Pexels absent (test+demo_eval) | 6 | 0.754 | 0.809 |
| F2 | Voxel51 present (val+test) | 142 | 0.998 | 0.999 |
| F5 | Pexels present (test+demo_eval) | 16 | 0.793 | 0.952 |
| F5 | Pexels absent (test+demo_eval) | 2 | 0.561 | 0.561 |
| F5 | Voxel51 present (val+test) | 146 | 1.000 | 1.000 |

### Shortcut

| factor | domain AUROC | n Voxel51 | n Pexels | val threshold | pos above | neg above | window AUROC | window pos/neg |
|---|---|---|---|---|---|---|---|---|
| F2 | 1.000 | 142 | 12 | 0.327 | 0.917 | 1.000 | 0.712 | 58/39 |
| F5 | 1.000 | 146 | 16 | 0.976 | 0.375 | 0.000 | 0.749 | 81/16 |

## ppe_ft_frz

Checkpoint `/mnt/c/projects/Meta Global AI Developer Hackathon/Sightline/training/04-head/checkpoints/ppe_ft_frz.pt`, hidden 512, dropout 0.2.

### Pexels clip metrics

| factor | split | pos | neg | AUROC | AP | AUROC 95% CI |
|---|---|---|---|---|---|---|
| F2 | test | 4 | 1 | 1.000 | 1.000 |  |
| F2 | demo_eval | 8 | 5 | 0.200 | 0.492 |  |
| F2 | held_out | 12 | 6 | 0.389 | 0.663 | [0.125, 0.728] |
| F2 | val | 8 | 9 | 0.764 | 0.734 |  |
| F2 | train | 29 | 17 | 0.779 | 0.819 |  |
| F5 | test | 5 | 0 | n/a | n/a |  |
| F5 | demo_eval | 11 | 2 | 0.636 | 0.933 |  |
| F5 | held_out | 16 | 2 | 0.750 | 0.969 | [0.511, 0.941] |
| F5 | val | 10 | 9 | 0.889 | 0.922 |  |
| F5 | train | 31 | 18 | 0.892 | 0.912 |  |

### Held-out score distributions

| factor | group | n | mean | median |
|---|---|---|---|---|
| F2 | Pexels present (test+demo_eval) | 12 | 0.618 | 0.607 |
| F2 | Pexels absent (test+demo_eval) | 6 | 0.644 | 0.639 |
| F2 | Voxel51 present (val+test) | 142 | 0.953 | 0.974 |
| F5 | Pexels present (test+demo_eval) | 16 | 0.706 | 0.796 |
| F5 | Pexels absent (test+demo_eval) | 2 | 0.632 | 0.632 |
| F5 | Voxel51 present (val+test) | 146 | 0.992 | 0.995 |

### Shortcut

| factor | domain AUROC | n Voxel51 | n Pexels | val threshold | pos above | neg above | window AUROC | window pos/neg |
|---|---|---|---|---|---|---|---|---|
| F2 | 1.000 | 142 | 12 | 0.577 | 0.667 | 0.833 | 0.196 | 58/39 |
| F5 | 1.000 | 146 | 16 | 0.883 | 0.125 | 0.000 | 0.736 | 81/16 |

## ppe_ft_frz_w05

Checkpoint `/mnt/c/projects/Meta Global AI Developer Hackathon/Sightline/training/04-head/checkpoints/ppe_ft_frz_w05.pt`, hidden 512, dropout 0.2.

### Pexels clip metrics

| factor | split | pos | neg | AUROC | AP | AUROC 95% CI |
|---|---|---|---|---|---|---|
| F2 | test | 4 | 1 | 1.000 | 1.000 |  |
| F2 | demo_eval | 8 | 5 | 0.050 | 0.450 |  |
| F2 | held_out | 12 | 6 | 0.194 | 0.530 | [0.000, 0.444] |
| F2 | val | 8 | 9 | 0.792 | 0.748 |  |
| F2 | train | 29 | 17 | 0.884 | 0.928 |  |
| F5 | test | 5 | 0 | n/a | n/a |  |
| F5 | demo_eval | 11 | 2 | 0.818 | 0.971 |  |
| F5 | held_out | 16 | 2 | 0.875 | 0.986 | [0.688, 1.000] |
| F5 | val | 10 | 9 | 0.889 | 0.899 |  |
| F5 | train | 31 | 18 | 0.950 | 0.953 |  |

### Held-out score distributions

| factor | group | n | mean | median |
|---|---|---|---|---|
| F2 | Pexels present (test+demo_eval) | 12 | 0.605 | 0.594 |
| F2 | Pexels absent (test+demo_eval) | 6 | 0.713 | 0.708 |
| F2 | Voxel51 present (val+test) | 142 | 0.910 | 0.949 |
| F5 | Pexels present (test+demo_eval) | 16 | 0.775 | 0.819 |
| F5 | Pexels absent (test+demo_eval) | 2 | 0.606 | 0.606 |
| F5 | Voxel51 present (val+test) | 146 | 0.996 | 0.997 |

### Shortcut

| factor | domain AUROC | n Voxel51 | n Pexels | val threshold | pos above | neg above | window AUROC | window pos/neg |
|---|---|---|---|---|---|---|---|---|
| F2 | 1.000 | 142 | 12 | 0.531 | 0.750 | 1.000 | 0.209 | 58/39 |
| F5 | 1.000 | 146 | 16 | 0.902 | 0.375 | 0.000 | 0.775 | 81/16 |

## ppe_pex_all

Checkpoint `/mnt/c/projects/Meta Global AI Developer Hackathon/Sightline/training/04-head/checkpoints/ppe_pex_all.pt`, hidden 512, dropout 0.2.

### Pexels clip metrics

| factor | split | pos | neg | AUROC | AP | AUROC 95% CI |
|---|---|---|---|---|---|---|
| F2 | test | 4 | 1 | 0.750 | 0.950 |  |
| F2 | demo_eval | 8 | 5 | 0.550 | 0.766 |  |
| F2 | held_out | 12 | 6 | 0.583 | 0.817 | [0.292, 0.850] |
| F2 | val | 8 | 9 | 0.597 | 0.577 |  |
| F2 | train | 29 | 17 | 1.000 | 1.000 |  |
| F5 | test | 5 | 0 | n/a | n/a |  |
| F5 | demo_eval | 11 | 2 | 0.727 | 0.952 |  |
| F5 | held_out | 16 | 2 | 0.812 | 0.977 | [0.600, 1.000] |
| F5 | val | 10 | 9 | 0.844 | 0.909 |  |
| F5 | train | 31 | 18 | 1.000 | 1.000 |  |

### Held-out score distributions

| factor | group | n | mean | median |
|---|---|---|---|---|
| F2 | Pexels present (test+demo_eval) | 12 | 0.735 | 0.868 |
| F2 | Pexels absent (test+demo_eval) | 6 | 0.753 | 0.849 |
| F2 | Voxel51 present (val+test) | 142 | 0.947 | 0.990 |
| F5 | Pexels present (test+demo_eval) | 16 | 0.785 | 0.930 |
| F5 | Pexels absent (test+demo_eval) | 2 | 0.483 | 0.483 |
| F5 | Voxel51 present (val+test) | 146 | 0.994 | 0.999 |

### Shortcut

| factor | domain AUROC | n Voxel51 | n Pexels | val threshold | pos above | neg above | window AUROC | window pos/neg |
|---|---|---|---|---|---|---|---|---|
| F2 | 0.813 | 142 | 12 | 0.195 | 0.917 | 1.000 | 0.708 | 58/39 |
| F5 | 0.946 | 146 | 16 | 0.984 | 0.375 | 0.000 | 0.822 | 81/16 |

## ppe_pex_frz

Checkpoint `/mnt/c/projects/Meta Global AI Developer Hackathon/Sightline/training/04-head/checkpoints/ppe_pex_frz.pt`, hidden 512, dropout 0.2.

### Pexels clip metrics

| factor | split | pos | neg | AUROC | AP | AUROC 95% CI |
|---|---|---|---|---|---|---|
| F2 | test | 4 | 1 | 1.000 | 1.000 |  |
| F2 | demo_eval | 8 | 5 | 0.025 | 0.444 |  |
| F2 | held_out | 12 | 6 | 0.194 | 0.525 | [0.000, 0.501] |
| F2 | val | 8 | 9 | 0.750 | 0.739 |  |
| F2 | train | 29 | 17 | 0.955 | 0.976 |  |
| F5 | test | 5 | 0 | n/a | n/a |  |
| F5 | demo_eval | 11 | 2 | 0.773 | 0.962 |  |
| F5 | held_out | 16 | 2 | 0.844 | 0.982 | [0.647, 1.000] |
| F5 | val | 10 | 9 | 0.833 | 0.857 |  |
| F5 | train | 31 | 18 | 0.952 | 0.964 |  |

### Held-out score distributions

| factor | group | n | mean | median |
|---|---|---|---|---|
| F2 | Pexels present (test+demo_eval) | 12 | 0.590 | 0.599 |
| F2 | Pexels absent (test+demo_eval) | 6 | 0.751 | 0.791 |
| F2 | Voxel51 present (val+test) | 142 | 0.637 | 0.626 |
| F5 | Pexels present (test+demo_eval) | 16 | 0.768 | 0.811 |
| F5 | Pexels absent (test+demo_eval) | 2 | 0.620 | 0.620 |
| F5 | Voxel51 present (val+test) | 146 | 0.988 | 0.990 |

### Shortcut

| factor | domain AUROC | n Voxel51 | n Pexels | val threshold | pos above | neg above | window AUROC | window pos/neg |
|---|---|---|---|---|---|---|---|---|
| F2 | 0.538 | 142 | 12 | 0.650 | 0.333 | 0.833 | 0.178 | 58/39 |
| F5 | 1.000 | 146 | 16 | 0.906 | 0.250 | 0.000 | 0.808 | 81/16 |

## ppe_sam_frz

Checkpoint `/mnt/c/projects/Meta Global AI Developer Hackathon/Sightline/training/04-head/checkpoints/ppe_sam_frz.pt`, hidden 512, dropout 0.2.

### Pexels clip metrics

| factor | split | pos | neg | AUROC | AP | AUROC 95% CI |
|---|---|---|---|---|---|---|
| F2 | test | 4 | 1 | 0.750 | 0.950 |  |
| F2 | demo_eval | 8 | 5 | 0.575 | 0.669 |  |
| F2 | held_out | 12 | 6 | 0.625 | 0.764 | [0.333, 0.911] |
| F2 | val | 8 | 9 | 0.833 | 0.852 |  |
| F2 | train | 29 | 17 | 0.890 | 0.941 |  |
| F5 | test | 5 | 0 | n/a | n/a |  |
| F5 | demo_eval | 11 | 2 | 0.182 | 0.751 |  |
| F5 | held_out | 16 | 2 | 0.250 | 0.841 | [0.000, 0.600] |
| F5 | val | 10 | 9 | 0.844 | 0.873 |  |
| F5 | train | 31 | 18 | 0.986 | 0.992 |  |

### Held-out score distributions

| factor | group | n | mean | median |
|---|---|---|---|---|
| F2 | Pexels present (test+demo_eval) | 12 | 0.665 | 0.682 |
| F2 | Pexels absent (test+demo_eval) | 6 | 0.591 | 0.632 |
| F2 | Voxel51 present (val+test) | 142 | 0.884 | 0.913 |
| F5 | Pexels present (test+demo_eval) | 16 | 0.618 | 0.625 |
| F5 | Pexels absent (test+demo_eval) | 2 | 0.781 | 0.781 |
| F5 | Voxel51 present (val+test) | 146 | 0.867 | 0.874 |

### Shortcut

| factor | domain AUROC | n Voxel51 | n Pexels | val threshold | pos above | neg above | window AUROC | window pos/neg |
|---|---|---|---|---|---|---|---|---|
| F2 | 0.914 | 142 | 12 | 0.477 | 0.833 | 0.667 | 0.539 | 58/39 |
| F5 | 0.862 | 146 | 16 | 0.478 | 0.812 | 1.000 | 0.276 | 81/16 |

## ppe_sam_shwd_all

Checkpoint `/mnt/c/projects/Meta Global AI Developer Hackathon/Sightline/training/04-head/checkpoints/ppe_sam_shwd_all.pt`, hidden 512, dropout 0.2.

### Pexels clip metrics

| factor | split | pos | neg | AUROC | AP | AUROC 95% CI |
|---|---|---|---|---|---|---|
| F2 | test | 4 | 1 | 1.000 | 1.000 |  |
| F2 | demo_eval | 8 | 5 | 0.750 | 0.860 |  |
| F2 | held_out | 12 | 6 | 0.792 | 0.899 | [0.549, 1.000] |
| F2 | val | 8 | 9 | 0.861 | 0.830 |  |
| F2 | train | 29 | 17 | 0.998 | 0.999 |  |
| F5 | test | 5 | 0 | n/a | n/a |  |
| F5 | demo_eval | 11 | 2 | 0.545 | 0.905 |  |
| F5 | held_out | 16 | 2 | 0.656 | 0.951 | [0.353, 0.941] |
| F5 | val | 10 | 9 | 0.944 | 0.956 |  |
| F5 | train | 31 | 18 | 1.000 | 1.000 |  |

### Held-out score distributions

| factor | group | n | mean | median |
|---|---|---|---|---|
| F2 | Pexels present (test+demo_eval) | 12 | 0.762 | 0.784 |
| F2 | Pexels absent (test+demo_eval) | 6 | 0.457 | 0.401 |
| F2 | Voxel51 present (val+test) | 142 | 0.886 | 0.944 |
| F5 | Pexels present (test+demo_eval) | 16 | 0.694 | 0.855 |
| F5 | Pexels absent (test+demo_eval) | 2 | 0.661 | 0.661 |
| F5 | Voxel51 present (val+test) | 146 | 0.958 | 0.991 |

### Shortcut

| factor | domain AUROC | n Voxel51 | n Pexels | val threshold | pos above | neg above | window AUROC | window pos/neg |
|---|---|---|---|---|---|---|---|---|
| F2 | 0.671 | 142 | 12 | 0.497 | 0.917 | 0.500 | 0.648 | 58/39 |
| F5 | 0.843 | 146 | 16 | 0.808 | 0.562 | 0.500 | 0.542 | 81/16 |

## ppe_sam_shwd_frz

Checkpoint `/mnt/c/projects/Meta Global AI Developer Hackathon/Sightline/training/04-head/checkpoints/ppe_sam_shwd_frz.pt`, hidden 512, dropout 0.2.

### Pexels clip metrics

| factor | split | pos | neg | AUROC | AP | AUROC 95% CI |
|---|---|---|---|---|---|---|
| F2 | test | 4 | 1 | 0.750 | 0.950 |  |
| F2 | demo_eval | 8 | 5 | 0.375 | 0.652 |  |
| F2 | held_out | 12 | 6 | 0.486 | 0.760 | [0.219, 0.750] |
| F2 | val | 8 | 9 | 0.542 | 0.632 |  |
| F2 | train | 29 | 17 | 0.848 | 0.923 |  |
| F5 | test | 5 | 0 | n/a | n/a |  |
| F5 | demo_eval | 11 | 2 | 0.318 | 0.837 |  |
| F5 | held_out | 16 | 2 | 0.375 | 0.888 | [0.062, 0.706] |
| F5 | val | 10 | 9 | 0.822 | 0.778 |  |
| F5 | train | 31 | 18 | 0.987 | 0.992 |  |

### Held-out score distributions

| factor | group | n | mean | median |
|---|---|---|---|---|
| F2 | Pexels present (test+demo_eval) | 12 | 0.477 | 0.443 |
| F2 | Pexels absent (test+demo_eval) | 6 | 0.462 | 0.485 |
| F2 | Voxel51 present (val+test) | 142 | 0.707 | 0.751 |
| F5 | Pexels present (test+demo_eval) | 16 | 0.627 | 0.700 |
| F5 | Pexels absent (test+demo_eval) | 2 | 0.788 | 0.788 |
| F5 | Voxel51 present (val+test) | 146 | 0.798 | 0.834 |

### Shortcut

| factor | domain AUROC | n Voxel51 | n Pexels | val threshold | pos above | neg above | window AUROC | window pos/neg |
|---|---|---|---|---|---|---|---|---|
| F2 | 0.815 | 142 | 12 | 0.154 | 1.000 | 1.000 | 0.469 | 58/39 |
| F5 | 0.701 | 146 | 16 | 0.651 | 0.562 | 1.000 | 0.290 | 81/16 |
