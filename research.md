了解。
では**研究計画書ではなく、手法仕様書**として書く。
これは「Codex に投げれば、何を実装すべきか迷わない」ことだけを目的にした文書だ。日程、工数、週ごとの進め方は一切入れない。中核は、前に整理した **structured locomotion memory / feasibility-aware retrieval / proposal-conditioned execution** に固定する。

---

# 手法仕様書

## 可実行性を考慮した運動記憶想起に基づくヒューマノイドの自然な移動制御

## 1. 文書の目的

本手法の目的は、ヒューマノイドの locomotion を

* 速度指令から直接関節行動を出すだけの policy
* 単一参照軌道を追従する tracking policy
* 単一 generator の出力をそのまま使う prior-based policy

としてではなく、

**構造化された locomotion memory から、現在状態・移動指令・接触位相に整合する短期 motion snippet を逐次選択し、それを proposal として execution policy に与える制御問題**

として定式化することである。
この文書は、そのために必要なモジュール、入出力、内部表現、数式、制約を固定する。

---

# 2. この手法で解く問題

既存の locomotion policy は、速度追従はできても、次の点で弱い。

1. **walk↔run transition が不自然になりやすい**
2. **同じ速度でも複数の歩容を自然に出し分けにくい**
3. **外乱や接触ズレがあると、自然さと安定性の両立が崩れる**

本手法はこれに対して、

* future motion を**その場で記憶から想起する**
* ただし単純最近傍ではなく、**可実行性で縛る**
* snippet は**厳密追従目標ではなく提案**として使う

という構造を採用する。
ここが核であり、単なる motion pool lookup ではない。

---

# 3. 採用する最終構成

システム全体は4モジュールで構成する。

## Module A. Structured Locomotion Memory

歩行・走行・旋回・遷移を含む motion 集合を、位相・接触・速度・歩容属性つきの snippet database に変換する。

## Module B. Feasibility-Aware Retriever

現在状態・指令・位相文脈から、次に使うべき短期 snippet を選ぶ。

## Module C. Proposal Encoder

retrieved snippet から、execution policy に渡す短期未来情報を抽出・整形する。

## Module D. Proposal-Conditioned Execution Policy

proposal を条件として受け取りつつ、実機制約・接触・外乱に応じて修正された行動を出力する。

---

# 4. 問題設定

## 4.1 入力

時刻 (t) における入力は以下。

### ロボット状態 (s_t)

* 関節角 (q_t \in \mathbb{R}^J)
* 関節角速度 (\dot{q}_t \in \mathbb{R}^J)
* base orientation
* base angular velocity
* base linear velocity estimate
* pelvis height
* foot contact state
* support foot state
* gait phase estimate

### 指令 (c_t)

* target forward velocity (v_x^{cmd})
* target lateral velocity (v_y^{cmd})
* target yaw rate (\omega_z^{cmd})

第一版では lateral はゼロ固定でもよい。
だがインターフェースには残す。

### 履歴 (h_t)

* 直前 contact state
* 直前 support foot
* 直前 snippet id またはその埋め込み
* 直前 gait type

---

# 5. Module A: Structured Locomotion Memory

ここが土台。
ここが雑なら retrieval も policy も全部死ぬ。

## 5.1 Source Motion

source は以下を想定する。

* Kimodo 等で生成した歩行・走行 motion
* 必要なら mocap 由来 motion
* humanoid 用 reference motion

重要なのは source の種類ではなく、**最終的に locomotion memory として使える形に落ちること**である。
Kimodo を使うこと自体は売りにならない。売りは memory の構造化だ。

---

## 5.2 Retarget 後の clip 表現

各 clip は最低限以下を持つ。

* root pose sequence
* root velocity sequence
* joint positions
* joint velocities
* left/right foot pose
* left/right foot velocity
* foot contact labels
* optional COM / pelvis / arm swing

フレームレートは固定する。
推奨は 30Hz か 60Hz。
第一版は 30Hz で十分。

---

## 5.3 Snippet 化

各 clip を固定長の短期区間に分割する。

### 推奨固定値

* snippet 長: **0.8秒**
* clip fps: **30Hz**
* 1 snippet の frame 数: **24**
* snippet stride: **6 frame**

つまり 0.2 秒刻みで新しい snippet を切り出す。

### 理由

* 短すぎると「未来運動」として意味がない
* 長すぎると online 切替の自由度が落ちる
* 0.8秒前後が「次の一歩〜数歩先」の提案として扱いやすい

---

## 5.4 Contact 再注釈

source 由来の contact は信用しすぎるな。
必ず再注釈する。

### contact 判定規則

各 foot について

* 足の高さ < (h_{contact})
* 足の世界座標速度 < (v_{contact})

を満たしたら接地候補とする。さらにヒステリシスを入れる。

### 初期値

* (h_{contact} = 0.03) m
* (v_{contact} = 0.15) m/s

### 出力

各 frame に対して

* left_contact ∈ {0,1}
* right_contact ∈ {0,1}
* support_type ∈ {left, right, double, flight}

を付与する。

---

## 5.5 Gait Phase 付与

各 frame に phase (\phi \in [0,1)) を付与する。

### 基本方針

* 左足接地開始イベントを 1周期の基準にする
* 連続するイベント間を線形補間して phase を振る
* run で接地様式が変わる場合も、support transition ベースで連続な phase を維持する

### 出力

各 snippet に対して

* phase_start
* phase_end
* phase trajectory

を持たせる。

---

## 5.6 Snippet 属性

各 snippet には最低限以下を持たせる。

```text
id
source_clip_id
start_frame
end_frame
duration
gait_type
transition_type
root_pose_seq
root_vel_seq
joint_pos_seq
joint_vel_seq
foot_contact_seq
phase_start
phase_end
mean_vx
mean_vy
mean_yaw_rate
cadence
step_length
duty_factor
com_height_mean
pelvis_bounce_amp
arm_swing_amp
quality_score
```

---

## 5.7 Gait Type

第一版で扱う gait type はこれだけでよい。

* walk
* run
* turn
* transition_walk_to_run
* transition_run_to_walk
* start
* stop

横歩きや後退は後で足せばいい。
最初から広げるな。

---

## 5.8 Transition Snippet

この研究で一番重要なのは transition。
だから transition は独立に扱う。

### 抽出条件

ある snippet を transition とみなす条件は、窓内で

* 平均速度が walk 域から run 域に跨ぐ
* cadence が増減している
* duty factor が変化している
* COM height が系統的に変化している

のいずれか、または複数。

### 意味

retriever が速度境界付近で transition snippet を選べるようにするため。
これがないと walk か run のどちらかに飛び、遷移が不自然になる。

---

## 5.9 Quality Filtering

低品質 snippet を memory に入れるな。

### 除外条件

* 関節 limit を大きく超える
* foot penetration が大きい
* contact と足速度が矛盾
* root drift が異常
* 骨盤高さが不自然に暴れる
* gait label が明らかに破綻

### 採用条件

* contact consistency が閾値以上
* quality score が閾値以上
* 遷移区間が連続的である

---

# 6. Module B: Feasibility-Aware Retriever

ここが研究の中心。
単純最近傍にした時点で新規性がほぼ死ぬ。
必要なのは**今その snippet が使えるか**まで入れた retrieval だ。

---

## 6.1 Query Feature

時刻 (t) における query ベクトル (x_q) を以下で構成する。

```text
cmd_vx
cmd_vy
cmd_yaw
base_lin_vel_est[3]
base_ang_vel_est[3]
pelvis_height
phase_sin
phase_cos
left_contact
right_contact
support_foot_onehot
prev_gait_onehot
prev_snippet_embedding(optional)
```

### 注意

phase は角度そのままではなく `sin/cos` で持つ。
support foot は left/right/double_or_flight の one-hot で持つ。

---

## 6.2 Memory Feature

各 snippet (m_i) の retrieval 用特徴 (x_i) は以下。

```text
mean_vx
mean_vy
mean_yaw_rate
phase_start_sin
phase_start_cos
phase_end_sin
phase_end_cos
left_contact_ratio
right_contact_ratio
duty_factor
cadence
step_length
com_height_mean
gait_onehot
transition_onehot
quality_score
```

---

## 6.3 Scoring Function

各 snippet に対して総合スコアを計算する。

[
S_i =
\alpha S_{\text{task}}

* \beta S_{\text{phase}}
* \gamma S_{\text{contact}}
* \delta S_{\text{connect}}
* \epsilon S_{\text{quality}}
* \zeta S_{\text{transition}}
  ]

---

### 6.3.1 Task Consistency

[
S_{\text{task}}
===============

*

\left(
w_v \left| \bar{v}*{xy}^{(i)} - v*{cmd,xy} \right|^2
+
w_{\omega} \left( \bar{\omega}_z^{(i)} - \omega_z^{cmd} \right)^2
\right)
]

意味:

* 指令速度・旋回率に合う snippet を優先する

---

### 6.3.2 Phase Continuity

[
S_{\text{phase}} = \cos(\phi_t - \phi^{(i)}_{start})
]

意味:

* 位相が合う snippet を優先
* 位相が飛ぶと歩容が急に崩れるため

---

### 6.3.3 Contact Compatibility

現在の接地状態と snippet 冒頭の接地状態が整合するかを見る。

### ルール例

* 現在 left support なのに snippet 冒頭が right-only contact なら大減点
* double support から walk/run 開始の transition は弱減点
* flight へ飛ぶ run transition は境界条件で許容

これは learned でなくていい。
最初はルールベースで十分。

---

### 6.3.4 Connection Feasibility

これが最重要。
単に似ている snippet ではなく、**今の姿勢から接続可能な snippet**を選ぶ。

[
S_{\text{connect}}
==================

*

## \left| q_t - q^{(i)}*0 \right|^2*{W_q}

## \left| \dot{q}*t - \dot{q}^{(i)}*0 \right|^2*{W*{\dot{q}}}

\left| r_t - r^{(i)}*0 \right|^2*{W_r}
]

ここで

* (q_t): 現在関節角
* (\dot{q}_t): 現在関節角速度
* (r_t): root/pelvis 周辺特徴

### 直感

見た目が合っていても、初期姿勢が全く違えば実行できない。
そこを罰する。

---

### 6.3.5 Quality Prior

[
S_{\text{quality}} = quality_score^{(i)}
]

低品質 snippet を選びにくくする。

---

### 6.3.6 Transition Bias

速度境界や gait 切替近傍では transition snippet を優先する。

例:

* (v_x^{cmd}) が walk/run 閾値近傍
* 直前 gait と現在指令が矛盾
* 直前 snippet が walk で現在 command が run 域

この時だけ transition prior を加点する。

---

## 6.4 初期重み

初期推奨値を固定しておく。

```text
alpha   = 1.0
beta    = 0.8
gamma   = 1.5
delta   = 1.2
epsilon = 0.3
zeta    = 0.7
```

理由:

* contact mismatch は強く罰する
* task は当然重要
* phase と connection はその次
* quality は prior 程度
* transition bias は常に効かせすぎない

---

## 6.5 Selection

第一版は **top-1 retrieval** に固定する。
top-K 平均や fancy aggregation は不要。むしろ害が出やすい。
locomotion で複数候補を雑に混ぜると、位相も接地も平均化されて気持ち悪い動きになる。
最初は top-1 で十分。

[
m_t^* = \arg\max_i S_i
]

---

## 6.6 Hysteresis

snippet の切替が多すぎると不自然になる。
なのでヒステリシスを入れる。

### ルール

候補 `m_new` が選ばれても、

[
S(m_{new}) > S(m_{cur}) + margin
]

を (K) 回連続で満たさない限り切り替えない。

### 推奨初期値

* `margin = 0.15`
* `K = 2`

---

## 6.7 Retrieval Frequency

* control frequency: **50Hz**
* retrieval frequency: **10Hz**

つまり 5 step ごとに retrieval を更新する。
毎 step 更新すると切替過剰になる。

---

# 7. Module C: Proposal Encoder

retrieved snippet をそのまま policy に投げるな。
情報が多すぎると、policy が proposal を無視するか、逆に rigid tracking に寄る。

必要なのは**未来の要点だけ**だ。

---

## 7.1 Proposal として渡す情報

現在 phase を基準に、未来時刻

[
\tau \in {0.2, 0.4, 0.6, 0.8}\text{ sec}
]

に対応するノードを抜く。

各未来ノードで policy に渡すものは以下。

* future root local displacement ((dx, dy, d\theta))
* future pelvis height
* future left/right contact
* optional reduced pose anchor

第一版では、
**root + contact + pelvis** で十分。
関節 full pose を丸ごと渡す必要はない。

---

## 7.2 Proposal 表現

proposal vector (p_t) の一例:

```text
for tau in [0.2, 0.4, 0.6, 0.8]:
    dx_tau
    dy_tau
    dyaw_tau
    pelvis_h_tau
    left_contact_tau
    right_contact_tau
```

必要なら最後に

* snippet gait type one-hot
* transition flag

を付ける。

---

# 8. Module D: Proposal-Conditioned Execution Policy

retrieved snippet を**proposal**として使い、policy はそれを身体制約の下で変形して実行する。
ここを hard tracking にした瞬間、研究の価値がかなり落ちる。
前の整理でも、**retrieved references as proposals, not targets** を明示していたはずで、ここは絶対に外すな。

---

## 8.1 Policy Observation

policy の観測 (o_t) は次で構成する。

### proprioception

* projected gravity or base orientation
* base angular velocity
* joint positions
* joint velocities
* previous action
* current contact state

### command

* cmd_vx
* cmd_vy
* cmd_yaw

### phase/history

* sin(phi), cos(phi)
* support foot
* recent contact history

### proposal

* proposal vector (p_t)

---

## 8.2 Policy Output

第一版では以下を採用する。

* 出力: residual joint target (\Delta q_t)

最終的な target は

[
q_t^{target}
============

q_t^{nominal}
+
q_t^{snippet}
+
\lambda_{res}\Delta q_t
]

で構成し、PD controller に渡す。

### 意味

* `nominal`: 基本姿勢
* `snippet`: retrieved proposal の現在対応 frame
* `residual`: 実行可能にするための修正

### なぜ torque 直出しにしないか

本研究の主張は retrieval 構造であって、低レベルトルク制御の難しさではない。
最初から torque 直出しにすると論点が散る。

---

## 8.3 Reward

総報酬は以下。

[
R =
\lambda_1 R_{\text{cmd}}

* \lambda_2 R_{\text{upright}}
* \lambda_3 R_{\text{proposal}}
* \lambda_4 R_{\text{contact}}

- \lambda_5 P_{\text{slip}}
- \lambda_6 P_{\text{energy}}
- \lambda_7 P_{\text{action-rate}}
  ]

---

### 8.3.1 Command Tracking Reward

[
R_{\text{cmd}}
==============

\exp
\left(
-k_v |v_{xy} - v_{xy}^{cmd}|^2
-k_{\omega}(\omega_z - \omega_z^{cmd})^2
\right)
]

---

### 8.3.2 Upright Reward

* torso 姿勢維持
* 骨盤が過度に傾かない
* base height が維持される

---

### 8.3.3 Proposal Alignment Reward

ここは**緩く**効かせる。
強くしすぎると strict tracking に戻る。

対象は

* root local displacement
* pelvis height
* selected contact event
* optional pose anchor

で十分。
全関節 imitation はやりすぎ。

---

### 8.3.4 Contact Reward / Penalty

* stance foot が滑らない
* swing foot が十分持ち上がる
* 不自然な contact 切替を抑える

---

### 8.3.5 Penalty

* foot slip
* energy / torque
* action rate
* joint limit
* stumble / impact

---

## 8.4 初期 reward 重み例

```text
lambda_cmd      = 3.0
lambda_upright  = 1.0
lambda_proposal = 0.8
lambda_contact  = 1.0
lambda_slip     = 1.5
lambda_energy   = 0.001
lambda_actrate  = 0.01
```

proposal を大きくしすぎるな。
proposal は「提案」であって「命令」ではない。

---

# 9. Runtime Loop

実行時の流れは以下。

## Step 1

現在状態 (s_t)、指令 (c_t)、履歴 (h_t) を取得する。

## Step 2

retrieval tick のタイミングなら、query を生成して snippet を選ぶ。

## Step 3

選ばれた snippet から proposal vector (p_t) を作る。

## Step 4

policy に ((o_t, p_t)) を入力し、residual action を出す。

## Step 5

`nominal + snippet + residual` から joint target を作り、低レベル制御器で実行する。

---

## 擬似コード

```python
if step % retrieval_interval == 0:
    query = build_query(state, command, history)
    scores = [score(query, snippet, state) for snippet in memory_db]
    best = argmax(scores)
    current_snippet = hysteresis_select(current_snippet, best, scores)

proposal = encode_proposal(current_snippet, current_phase)
obs = build_obs(state, command, history, proposal)
delta_q = policy(obs)

q_target = q_nominal + q_ref_from_snippet(current_snippet, current_phase) + lambda_res * delta_q
action = pd_control(q_target)
```

---

# 10. 絶対に入れるべきベースライン

実装は Codex に任せるとしても、比較対象を最初から固定しておかないと手法がぼやける。

## Baseline A: Command-Conditioned Policy

速度指令だけで歩く普通の locomotion policy。

## Baseline B: Fixed Reference Tracking

事前に決めた gait reference を tracking する policy。

## Baseline C: Retrieval-Only Hard Tracking

retrieved snippet をそのまま追従する policy。

### なぜ必要か

提案手法は「retrieval したこと」ではなく、**proposal-conditioned execution** で勝たなければ意味がない。
だから `retrieval-only hard tracking` は外せない。

---

# 11. 絶対に入れるべきアブレーション

## Ablation 1

structured memory なし

## Ablation 2

phase score なし

## Ablation 3

contact compatibility なし

## Ablation 4

connection feasibility なし

## Ablation 5

proposal-conditioned ではなく hard tracking

## Ablation 6

transition snippet なし

この6つを消して差が出ないなら、正直その手法は弱い。

---

# 12. この手法で主に見るべき指標

本手法は「人間っぽい動画」でごまかしてはいけない。
見るべき軸は4つ。

## 12.1 タスク性能

* velocity RMSE
* yaw-rate RMSE
* success rate
* fall rate

## 12.2 自然さ

* cadence continuity
* stride smoothness
* pelvis height jerk
* transition jerk

## 12.3 実行可能性

* foot slip distance
* stumble count
* impact peak
* cost of transport

## 12.4 記憶利用の妥当性

* retrieval switch frequency
* contact-valid retrieval rate
* phase-valid retrieval rate
* transition snippet recall

この最後の軸が重要。
memory を使ったこと自体の意味を示せないなら、ただ複雑にしただけになる。

---

# 13. この手法で最も強い実験設定

手法の顔は明確にこれ。

## walk-to-run / run-to-walk transition

理由:

* fixed reference が弱い
* command-only では歩容の自然さが出にくい
* transition snippet の価値が最も見える

次点で強いのはこれ。

## same velocity, different style

同じ (1.5) m/s でも

* relaxed walk
* brisk walk
* near-jog

を出せるか。
ただし style latent を最初から大げさにやるな。
第一版は memory cluster の選択だけで十分。

---

# 14. この手法でやってはいけないこと

ここを外すと一気に弱くなる。

## 14.1 Motion pool の量を売りにすること

それは研究の芯ではない。
芯は memory の構造化と retrieval の可実行性だ。

## 14.2 top-K average を最初から入れること

大抵壊れる。
位相と接地が平均化されて終わる。

## 14.3 proposal を hard target にすること

それならただの tracking に戻る。

## 14.4 phase/contact を軽視すること

locomotion はここを外すと全部崩れる。

## 14.5 evaluation を tracking error だけにすること

自然さと実行可能性が抜けたら、この研究の意味がない。

---

# 15. この文書を読んだ Codex が理解すべきこと

Codex に判断させる部分はあっていい。
だが、**手法の自由度**はここで潰しておくべきだ。

Codex が勝手に変えてはいけないのは以下。

1. **memory は構造化する**
2. **retrieval は可実行性で縛る**
3. **selection は初版 top-1**
4. **snippet は proposal であり hard target ではない**
5. **主戦場は walk-run transition**
6. **評価は task / naturalness / executability / memory validity の4軸**

この6本は固定。
ここを Codex に好き勝手やらせると、すぐ普通の imitation tracking か command-conditioned RL に戻る。

---

# 16. 最終定義

この手法を一文で定義するとこうなる。

**本手法は、ヒューマノイド locomotion を単一参照の追従としてではなく、構造化された locomotion memory から可実行な短期運動 snippet を逐次選択し、それを proposal として execution policy が変形・実行する問題として解く。**

これが全てだ。
これ以上広げるな。
広げた瞬間に、また「面白そうな構想」に戻る。

必要なら次に、これをそのまま **Codex に渡すための“仕様書テンプレート”** にして、
`Goal / Inputs / Outputs / Constraints / Core Equations / Non-goals` の形に再整形する。

