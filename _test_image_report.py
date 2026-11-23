"""追加: 汇总图表 + HTML 报告（由 _test_image_modules.py 导入执行）"""
import os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

OUT = "reports/image_modules"
FIG = os.path.join(OUT, "figs")
THR = 0.5


def build(records, ckpt_meta):
    n = len(records)
    pf = np.array([r["p_fake"] for r in records])
    pred = np.array([r["pred"] for r in records])
    gt = np.array([r["gt"] for r in records])
    names = [r["name"] for r in records]

    acc = float((pred == gt).mean())
    tp = int(((pred == 1) & (gt == 1)).sum())
    tn = int(((pred == 0) & (gt == 0)).sum())
    fp = int(((pred == 1) & (gt == 0)).sum())
    fn = int(((pred == 0) & (gt == 1)).sum())
    rec_r = tn / max(tn + fp, 1)
    rec_f = tp / max(tp + fn, 1)
    bacc = (rec_r + rec_f) / 2

    # ---- 图1: p_fake 排序条 + 混淆矩阵 ----
    fig, axes = plt.subplots(1, 2, figsize=(19, 6.5),
                             gridspec_kw={"width_ratios": [2.4, 1]})
    order = np.argsort(-pf)
    colors = ["#e53935" if gt[i] == 1 else "#43a047" for i in order]
    ax = axes[0]
    ax.bar(range(n), pf[order], color=colors, edgecolor="k", linewidth=0.4)
    ax.axhline(THR, ls="--", c="k", lw=1.2)
    ax.text(n - 0.5, THR + 0.015, f"阈值 {THR}", ha="right", fontsize=9)
    ax.set_xticks(range(n))
    ax.set_xticklabels([names[i] for i in order], rotation=90, fontsize=7)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("p_fake（AI 生成概率）")
    ax.set_title("各测试图 p_fake（绿=启发式真实，红=启发式伪造）", fontsize=12)
    ax.grid(axis="y", alpha=0.3)

    ax = axes[1]
    cm = np.array([[tn, fp], [fn, tp]])
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["预测 Real", "预测 Fake"])
    ax.set_yticklabels(["真实 Real", "真实 Fake"])
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=20,
                    color="w" if cm[i, j] > cm.max() / 2 else "k")
    ax.set_title(f"混淆矩阵  acc={acc*100:.1f}%  bacc={bacc*100:.1f}%", fontsize=12)
    plt.colorbar(im, ax=ax, fraction=0.046)
    plt.tight_layout()
    fig.savefig(os.path.join(FIG, "summary_overview.png"), dpi=110, bbox_inches="tight")
    plt.close(fig)

    # ---- 图2: MSFF 权重 + 特征统计 ----
    W = np.array([r["msff_w"] for r in records])
    fig, axes = plt.subplots(1, 3, figsize=(20, 5.8))

    ax = axes[0]
    bottom = np.zeros(n)
    for k, (lab, col) in enumerate(zip(["L3 浅层", "L6 中层", "L12 深层"],
                                       ["#7986cb", "#4fc3f7", "#4db6ac"])):
        ax.bar(range(n), W[:, k], bottom=bottom, label=lab, color=col)
        bottom += W[:, k]
    ax.set_xticks(range(n))
    ax.set_xticklabels(names, rotation=90, fontsize=7)
    ax.set_ylabel("MSFF 尺度权重")
    ax.set_title("MSFF 多尺度注意力权重（逐图）", fontsize=12)
    ax.legend(fontsize=9, loc="lower right")
    ax.set_ylim(0, 1.02)

    ax = axes[1]
    data = [W[:, 0], W[:, 1], W[:, 2]]
    bp = ax.boxplot(data, tick_labels=["L3", "L6", "L12"], patch_artist=True, widths=0.55)
    for patch, c in zip(bp["boxes"], ["#7986cb", "#4fc3f7", "#4db6ac"]):
        patch.set_facecolor(c)
    ax.set_ylabel("权重")
    ax.set_title("MSFF 尺度权重分布", fontsize=12)
    ax.grid(axis="y", alpha=0.3)

    ax = axes[2]
    ng = [r["n_glob"] for r in records]
    nf = [r["n_fore"] for r in records]
    cg = [r["cos_gl"] for r in records]
    xpos = np.arange(n)
    ax.bar(xpos - 0.2, ng, width=0.4, label="‖全局‖", color="#5c6bc0")
    ax.bar(xpos + 0.2, nf, width=0.4, label="‖取证‖", color="#26a69a")
    ax2 = ax.twinx()
    ax2.plot(xpos, cg, "o-", ms=3, color="#e53935", lw=1, label="cos(全局,取证)")
    ax2.set_ylim(-1, 1)
    ax2.set_ylabel("cos(全局, 取证)", color="#e53935")
    ax.set_xticks(range(n))
    ax.set_xticklabels(names, rotation=90, fontsize=7)
    ax.set_ylabel("特征 L2 模长")
    ax.set_title("全局 / 取证 特征模长及一致性", fontsize=12)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=8, loc="upper left")
    plt.tight_layout()
    fig.savefig(os.path.join(FIG, "summary_msff_feats.png"), dpi=110, bbox_inches="tight")
    plt.close(fig)

    # ---- 图3: 取证统计 real vs fake ----
    keys = [("srm_std", "SRM 残差标准差"), ("srm_absmean", "SRM 残差绝对均值"),
            ("hf_ratio", "FFT 高频能量占比"), ("phase_std", "FFT 相位标准差")]
    fig, axes = plt.subplots(1, 4, figsize=(20, 4.8))
    for ax, (k, lab) in zip(axes, keys):
        vr = [r[k] for r in records if r["gt"] == 0]
        vf = [r[k] for r in records if r["gt"] == 1]
        bp = ax.boxplot([vr, vf], tick_labels=[f"Real(n={len(vr)})", f"Fake(n={len(vf)})"],
                        patch_artist=True, widths=0.5)
        for patch, c in zip(bp["boxes"], ["#43a047", "#e53935"]):
            patch.set_facecolor(c)
        ax.scatter([1] * len(vr), vr, s=18, c="k", alpha=0.5, zorder=3)
        ax.scatter([2] * len(vf), vf, s=18, c="k", alpha=0.5, zorder=3)
        ax.set_title(lab, fontsize=11)
        ax.grid(axis="y", alpha=0.3)
    plt.suptitle("局部取证支路统计量：真实 vs 伪造", fontsize=13)
    plt.tight_layout()
    fig.savefig(os.path.join(FIG, "summary_forensics.png"), dpi=110, bbox_inches="tight")
    plt.close(fig)

    # ---- HTML ----
    rows = []
    for r in sorted(records, key=lambda x: -x["p_fake"]):
        ok = (r["pred"] == r["gt"])
        mark = "✅" if ok else "❌"
        rows.append(f"""
        <tr class="{'ok' if ok else 'bad'}">
          <td><a href="figs/{r['fig']}" target="_blank">{r['name']}</a></td>
          <td>{'Fake' if r['gt']==1 else 'Real'}</td>
          <td class="num">{r['p_fake']:.4f}</td>
          <td>{'AI生成' if r['pred']==1 else '真实'}</td>
          <td>{mark}</td>
          <td class="num">{r['msff_w'][0]:.2f}/{r['msff_w'][1]:.2f}/{r['msff_w'][2]:.2f}</td>
          <td class="num">{r['cos_gl']:.3f}</td>
          <td class="num">{r['n_glob']:.1f}</td>
          <td class="num">{r['n_fore']:.1f}</td>
          <td class="num">{r['hf_ratio']:.3f}</td>
          <td><img class="thumb" src="figs/{r['fig']}" loading="lazy"></td>
        </tr>""")

    mean_w = W.mean(axis=0)

    # 其他图像权重基线（可选）
    base_rows = ""
    base_path = os.path.join(OUT, "baselines.json")
    if os.path.exists(base_path):
        try:
            with open(base_path, encoding="utf-8") as f:
                bl = json.load(f)
            trs = "".join(
                f"<tr><td>{b['model']}</td><td class='num'>{b['thr']:.2f}</td>"
                f"<td class='num'>{b['acc']*100:.1f}%</td>"
                f"<td class='num'>{b['bacc']*100:.1f}%</td>"
                f"<td class='num'>{b['rec_real']*100:.1f}%</td>"
                f"<td class='num'>{b['rec_fake']*100:.1f}%</td></tr>" for b in bl)
            base_rows = f"""
<h2>同测试集上的其他图像权重对比</h2>
<table><thead><tr><th>权重</th><th>阈值</th><th>Acc</th><th>BAcc</th><th>真实召回</th><th>伪造召回</th></tr></thead>
<tbody>{trs}
<tr style="background:#eef4ff;font-weight:600"><td>finetuned.pt（本报告主模型，CLIP+MSFF+取证）</td>
<td class='num'>{THR:.2f}</td><td class='num'>{acc*100:.1f}%</td><td class='num'>{bacc*100:.1f}%</td>
<td class='num'>{rec_r*100:.1f}%</td><td class='num'>{rec_f*100:.1f}%</td></tr>
</tbody></table>
<div class="sub">DeiT-Small 系列为 ImageDetectorViT（backbone+classifier 两模块），结果存于 baselines.json。</div>"""
        except Exception as e:
            base_rows = f"<div class='sub'>baselines.json 读取失败: {e}</div>"

    html = f"""<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8">
<title>test/image 全模块测试可视化</title>
<style>
 body{{font-family:"Microsoft YaHei",system-ui,sans-serif;margin:0;background:#f5f6f8;color:#1f2430}}
 .wrap{{max-width:1500px;margin:0 auto;padding:28px 22px 60px}}
 h1{{font-size:26px;margin:0 0 6px}} h2{{font-size:19px;margin:34px 0 12px;border-left:5px solid #4c6ef5;padding-left:10px}}
 .sub{{color:#6b7280;font-size:13px;margin-bottom:18px}}
 .cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(178px,1fr));gap:12px;margin:16px 0 8px}}
 .card{{background:#fff;border-radius:10px;padding:14px 16px;box-shadow:0 1px 3px rgba(0,0,0,.08)}}
 .card .k{{font-size:12px;color:#6b7280}} .card .v{{font-size:23px;font-weight:700;margin-top:5px}}
 .warn{{background:#fff7e6;border:1px solid #ffd591;border-radius:9px;padding:13px 16px;font-size:13.5px;line-height:1.75;margin:14px 0}}
 .note{{background:#eef4ff;border:1px solid #c7d8ff;border-radius:9px;padding:13px 16px;font-size:13.5px;line-height:1.75;margin:14px 0}}
 table{{width:100%;border-collapse:collapse;background:#fff;border-radius:10px;overflow:hidden;font-size:13px;box-shadow:0 1px 3px rgba(0,0,0,.08)}}
 th{{background:#eef1f6;padding:9px 8px;text-align:left;font-weight:600;font-size:12.5px;white-space:nowrap}}
 td{{padding:8px;border-top:1px solid #eef1f6;vertical-align:middle}}
 tr.bad td:first-child{{color:#c62828}} .num{{text-align:right;font-variant-numeric:tabular-nums}}
 img.thumb{{width:230px;border-radius:6px;display:block}}
 img.big{{width:100%;border-radius:10px;box-shadow:0 1px 4px rgba(0,0,0,.1);margin:8px 0}}
 a{{color:#3b5bdb;text-decoration:none}} a:hover{{text-decoration:underline}}
 .ok{{background:#fbfefb}} .bad{{background:#fffafa}}
</style></head><body><div class="wrap">
<h1>test/image 全模块测试可视化</h1>
<div class="sub">模型：ImageBranch（CLIP ViT-B/16 + MSFF + SRM/FFT 局部取证 + G-Fuse）&nbsp;|&nbsp;
权重：{os.path.basename('checkpoints/finetuned/finetuned.pt')}（epoch {ckpt_meta.get('epoch')}，训练 val_acc={ckpt_meta.get('val_acc'):.4f}）&nbsp;|&nbsp;
样本数：{n}&nbsp;|&nbsp;阈值：{THR}</div>

<div class="cards">
  <div class="card"><div class="k">准确率 Acc</div><div class="v">{acc*100:.1f}%</div></div>
  <div class="card"><div class="k">平衡准确率 BAcc</div><div class="v">{bacc*100:.1f}%</div></div>
  <div class="card"><div class="k">真实召回</div><div class="v">{rec_r*100:.1f}%</div></div>
  <div class="card"><div class="k">伪造召回</div><div class="v">{rec_f*100:.1f}%</div></div>
  <div class="card"><div class="k">混淆矩阵 TP/TN/FP/FN</div><div class="v" style="font-size:17px">{tp} / {tn} / {fp} / {fn}</div></div>
  <div class="card"><div class="k">MSFF 平均权重 L3/L6/L12</div><div class="v" style="font-size:16px">{mean_w[0]:.2f} / {mean_w[1]:.2f} / {mean_w[2]:.2f}</div></div>
</div>

<div class="warn"><b>⚠ 结论</b><br>
该 <code>finetuned.pt</code> 在 test/image 上严重失效：平衡准确率 {bacc*100:.1f}%，真实类召回 {rec_r*100:.1f}%。
绝大多数图像输出 p_fake≈1.0，说明分类头把「真实照片」也推向了「AI 生成」一侧，属于典型的分布偏移 + 过拟合（训练时 Real 类仅用 Picsum 场景图，且验证集与训练集同源）。
下方逐图面板可直观看到：判定几乎恒定在高分侧，MSFF 尺度权重与取证统计量的分布也高度集中。</div>

<div class="note"><b>模块说明</b><br>
① 原图 → ② CLIP 预处理 → ③ CLIP 注意力 Rollout → ④ 图像分类头输出。<br>
⑤ SRM 3×3 高频残差核（3 核 × 3 通道）→ ⑥ SRM 1×1 卷积投影到 16 通道 → ⑦ FFT 对数幅度谱 → ⑧ FFT 相位谱 → 这部分构成 <b>局部取证支路 Φ_forensics</b>。<br>
⑨ MSFF 对 CLIP 第 3/6/12 层 CLS 特征的层级注意力权重 → ⑩ 三层特征余弦相似度 → ⑪ 全局/取证特征模长与一致性 → ⑫ G-Fuse 门控权重与模态可靠性。<br>
<b>注：</b>⑫ 的 G-Fuse / X-Cons 权重未包含在该 checkpoint 中，为随机初始化，仅用于展示结构，输出不具备判别意义。</div>

<h2>汇总图表</h2>
<img class="big" src="figs/summary_overview.png">
<img class="big" src="figs/summary_msff_feats.png">
<img class="big" src="figs/summary_forensics.png">
{base_rows}

<h2>逐图模块面板（点击文件名查看大图）</h2>
<table>
<thead><tr>
 <th>文件</th><th>启发式GT</th><th>p_fake</th><th>模型判定</th><th>对错</th>
 <th>MSFF L3/L6/L12</th><th>cos(全局,取证)</th><th>‖全局‖</th><th>‖取证‖</th><th>FFT高频占比</th><th>模块面板</th>
</tr></thead>
<tbody>{''.join(rows)}</tbody>
</table>
<div class="sub" style="margin-top:16px">
生成脚本：_test_image_modules.py &nbsp;|&nbsp; 图像分辨率统一 224×224；注意力图尺寸 14×14 上采样显示。
启发式 GT 依据 calibration.json 的 <code>ground_truth.image_rule</code>（文件名 <code>real_</code> 前缀 = Real，其余 = Fake）。
</div>
</div></body></html>"""

    with open(os.path.join(OUT, "index.html"), "w", encoding="utf-8") as f:
        f.write(html)
    with open(os.path.join(OUT, "records.json"), "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=1)

    print(f"HTML: {OUT}/index.html")
    print(f"acc={acc*100:.2f}% bacc={bacc*100:.2f}% rec_real={rec_r*100:.1f}% "
          f"rec_fake={rec_f*100:.1f}%  TP={tp} TN={tn} FP={fp} FN={fn}")
    return dict(acc=acc, bacc=bacc, rec_real=rec_r, rec_fake=rec_f,
                tp=tp, tn=tn, fp=fp, fn=fn)
