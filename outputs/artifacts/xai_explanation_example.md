# Example XAI Explanation Output

## Instance: Sample Player Position #42

### Input Features
| Feature | Value | Description |
|---------|-------|-------------|
| x_position | 45.2 | Yards from end zone |
| y_position | 26.8 | Yards from sideline |
| speed | 7.3 | Yards per second |
| acceleration | 1.2 | Yards/second² |
| direction | 127.5 | Movement direction (degrees) |
| distance_to_ball | 3.8 | Yards from ball |
| time_to_ball | 0.52 | Estimated seconds |
| relative_speed | 2.1 | Speed relative to ball |
| angle_to_ball | 45.3 | Degrees |
| defender_proximity | 5.2 | Nearest defender distance |
| receiver_separation | 2.8 | Separation from coverage |
| play_time | 3.2 | Seconds since snap |

### Model Prediction
- **Prediction**: Catch Attempt (Positive Class)
- **Probability**: 0.73
- **Confidence**: High (MC Dropout std = 0.08)

---

## SHAP Explanation

### Global Feature Importance (SHAP Mean |Value|)
```
1. distance_to_ball     ████████████████████ 0.182
2. receiver_separation  ██████████████████   0.165
3. speed                ███████████████      0.142
4. time_to_ball         ████████████         0.118
5. angle_to_ball        ██████████           0.095
6. defender_proximity   ████████             0.078
7. relative_speed       ██████               0.062
8. x_position           █████                0.048
9. direction            ████                 0.041
10. y_position          ███                  0.032
11. acceleration        ██                   0.025
12. play_time           █                    0.012
```

### Local SHAP Values (This Instance)
```
Base value: 0.45 (average prediction)

Feature contributions to prediction (0.73):
  + distance_to_ball (3.8):      +0.12  ▲ Close to ball increases catch prob
  + receiver_separation (2.8):   +0.09  ▲ Good separation from defender
  + speed (7.3):                 +0.05  ▲ High speed toward ball
  - direction (127.5):           -0.02  ▼ Angle slightly suboptimal
  + time_to_ball (0.52):         +0.04  ▲ Quick arrival time

Final prediction: 0.45 + 0.28 = 0.73
```

### SHAP Force Plot (Text Representation)
```
              ◄───────── Lower ─────────     ─────────── Higher ──────────►

0.0          0.2          0.4          0.6          0.8          1.0
 ├────────────┼────────────┼────────────┼────────────┼────────────┤
                           │      ████████████████████│
                      Base │            Prediction    │
                      0.45                           0.73

                           distance_to_ball ▲▲▲▲
                           receiver_separation ▲▲▲
                           speed ▲▲
                           direction ▼
```

---

## LIME Explanation

### Local Linear Model
- **Local Model R²**: 0.91 (good local fidelity)
- **Intercept**: 0.42

### Feature Rules (Discretized)
| Rule | Weight | Interpretation |
|------|--------|----------------|
| distance_to_ball <= 5.0 | +0.18 | Close to ball → catch likely |
| receiver_separation > 2.0 | +0.12 | Open receiver → catch likely |
| speed > 6.5 | +0.08 | Fast player → catch likely |
| time_to_ball <= 0.75 | +0.06 | Quick arrival → catch likely |
| direction ∈ [120, 150] | -0.03 | Suboptimal angle |
| defender_proximity > 4.0 | +0.04 | No tight coverage |

### LIME Prediction Breakdown
```
Intercept:                    0.42
+ distance_to_ball <= 5.0:   +0.18
+ receiver_separation > 2.0: +0.12
+ speed > 6.5:               +0.08
+ time_to_ball <= 0.75:      +0.06
+ defender_proximity > 4.0:  +0.04
- direction ∈ [120, 150]:    -0.03
─────────────────────────────────────
Local Prediction:             0.87
Actual Model Prediction:      0.73
```

---

## Agreement Analysis

### SHAP vs LIME Top Features
| Rank | SHAP | LIME | Agreement |
|------|------|------|-----------|
| 1 | distance_to_ball | distance_to_ball | ✓ |
| 2 | receiver_separation | receiver_separation | ✓ |
| 3 | speed | speed | ✓ |
| 4 | time_to_ball | time_to_ball | ✓ |
| 5 | angle_to_ball | defender_proximity | ✗ |

**Overlap**: 4/5 top features agree (80%)

### Interpretation Consistency
- Both methods identify **distance_to_ball** as most important
- Both agree on positive contribution of **speed** and **separation**
- Minor disagreement on 5th feature (angle vs proximity)
- **Conclusion**: High consistency between methods

---

## Human-Readable Summary

### Why did the model predict "Catch Attempt"?

> The model predicts this player will attempt to catch the ball (73% probability) primarily because:
>
> 1. **Close to ball** (3.8 yards): The player's proximity to the ball landing spot strongly suggests they're the intended receiver.
>
> 2. **Good separation** (2.8 yards): The receiver has created sufficient separation from the nearest defender, increasing catch opportunity.
>
> 3. **High speed** (7.3 yd/s): The player is moving quickly toward the ball, indicating active pursuit.
>
> 4. **Quick arrival** (0.52 seconds): Low time-to-ball suggests the player is well-positioned.
>
> **Confidence Note**: The MC Dropout uncertainty (std=0.08) indicates the model is confident in this prediction. No human review required.

---

## Recommendation for Deployment

| Confidence Level | Threshold | Action |
|------------------|-----------|--------|
| High | prob > 0.7, std < 0.15 | Auto-approve |
| Medium | prob ∈ [0.4, 0.7] | Flag for review |
| Low | std > 0.2 | Require human decision |

**This instance**: High confidence → Auto-approve

---

*Generated: 2025-12-07*
*Model: DecisionTree (Classical ML)*
*XAI Methods: SHAP TreeExplainer, LIME TabularExplainer*
