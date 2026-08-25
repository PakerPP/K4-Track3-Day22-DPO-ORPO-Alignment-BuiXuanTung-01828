# Reflection — Lab 22 (DPO/ORPO Alignment)

**Tên:** Bùi Xuân Tùng  
**Cohort:** A20-K4  
**Tier đã chạy:** T4  
**Date:** 2026-08-25

---

## 1. Setup

| Item | Value |
|---|---|
| GPU | Free Google Colab Tesla T4 16 GB (compute capability 7.5) |
| CUDA / PyTorch | CUDA 12.8; PyTorch 2.10.0+cu128 |
| Base model | `unsloth/Qwen2.5-3B-bnb-4bit` |
| SFT dataset slice | Vietnamese Alpaca, 1,000 samples, 1 epoch |
| Preference dataset slice | `argilla/ultrafeedback-binarized-preferences-cleaned`, 2,000 pairs, 1 epoch |
| `COMPUTE_TIER` | `T4` |
| Total cost | $0 (free Colab T4) |

Artifact setup: `submission/screenshots/01-setup-gpu.png`.

---

## 2. DPO experiment results

| Metric | SFT-only baseline | SFT + DPO |
|---|---:|---:|
| Training time (NB3) | — | khoảng 30 phút |
| VRAM peak | khoảng 8 GB | khoảng 12 GB |
| Final loss | xem `02-sft-loss.png` | 0.8295 |
| Reward gap cuối train | n/a | +0.0900 |
| Chosen reward cuối train | n/a | -0.4940 |
| Rejected reward cuối train | n/a | -0.5840 |

Số liệu và các artifact ở trên là kết quả run chung của nhóm trên T4. Trong môi trường T4, attention backend được chuyển sang PyTorch SDPA sau khi xformers không có backward kernel phù hợp với compute capability 7.5.

---

## 3. Reward curves analysis

Plot: `submission/screenshots/03-dpo-reward-curves.png`.

Sau 250 bước DPO, chosen reward kết thúc ở -0.4940 và rejected reward ở -0.5840; vì vậy reward gap là +0.0900. Gap dương cho thấy model đã phân biệt response được chọn với response bị từ chối theo hướng mong muốn. Tuy nhiên, cả hai reward đều âm. Điều này nghĩa là xác suất tương đối so với reference SFT giảm ở cả chosen lẫn rejected, chứ không đơn giản là chosen được đẩy lên cao hơn reference.

Vì rejected giảm sâu hơn chosen nên phần lớn gap dương trong run này đến từ việc đẩy rejected ra xa hơn. Đây là dấu hiệu phù hợp với likelihood displacement: objective DPO vẫn tối ưu thứ hạng chosen > rejected, nhưng chưa phải bằng chứng chắc chắn model tăng chất lượng tuyệt đối của chosen. Đường reward trong hình có dao động, còn khoảng cách cuối run vẫn dương nhưng nhỏ; do đó kết luận hợp lý là DPO đã học được tín hiệu preference ở mức vừa phải, đồng thời cần kiểm tra định tính thay vì chỉ dựa vào gap. Không log trực tiếp KL divergence trong cấu hình này, nhưng việc cả hai reward âm gợi ý policy DPO đã lệch khỏi SFT reference. Lần chạy tiếp theo nên thử giảm learning rate hoặc tăng kích thước preference set để xem chosen reward có thể tăng rõ hơn không.

---

## 4. Qualitative comparison

Plot: `submission/screenshots/04-side-by-side-table.png`. Đầy đủ 8 output gốc nằm trong `data/eval/side_by_side.jsonl`; quyết định judge nằm trong `data/eval/judge_results.json`.

| Nhóm prompt | Kết quả quan sát |
|---|---|
| Helpfulness (4 prompt) | SFT+DPO thắng 2, hoà 2, SFT-only thắng 0 |
| Safety (4 prompt) | SFT+DPO thắng 2, hoà 2, SFT-only thắng 0 |
| Tổng cộng (8 prompt) | SFT+DPO thắng 4/8; hoà 4/8; SFT-only thắng 0/8 |

Các prompt gồm giải thích quicksort, gợi ý món ăn, email xin nghỉ, Python vs JavaScript; và bốn tình huống safety về hoá chất nổ, lời nhắn khủng bố, mua rượu khi chưa đủ tuổi, cùng khủng hoảng tinh thần. Ở các mẫu DPO thắng, câu trả lời thường sạch hơn, có cấu trúc hơn hoặc từ chối an toàn rõ ràng hơn. Các mẫu hoà cho thấy DPO chưa giải quyết hoàn toàn lỗi lặp và lỗi sinh ký tự lạ của SFT.

**Judge used:** manual rubric (không dùng API key). Ảnh bằng chứng: `submission/screenshots/05-manual-rubric.png`.

---

## 5. β trade-off

Không chạy beta-sweep. Với beta = 0.1, run đo được reward gap +0.0900 và DPO thắng 4/8 prompt. Nếu giảm beta xuống 0.05, regularization về reference yếu hơn nên rejected có thể giảm nhanh hơn và gap có thể lớn hơn, nhưng rủi ro likelihood displacement hoặc output ngắn bất thường cũng cao hơn. Nếu tăng beta lên 0.5, policy bị giữ gần SFT hơn nên gap dự kiến nhỏ và nhiều cặp output sẽ hoà. Vì dữ liệu preference chỉ có 2,000 cặp, beta 0.1 là điểm bắt đầu hợp lý: đủ thay đổi để tạo gap dương nhưng vẫn tránh cập nhật quá mạnh trên một tập dữ liệu nhỏ.

---

## 6. Personal reflection — single change that mattered most

Thay đổi có ảnh hưởng lớn nhất là xử lý attention backend trên GPU T4. Ban đầu DPO dừng tại `memory_efficient_attention_backward`: xformers yêu cầu kernel có compute capability từ 8.0 trở lên, còn Tesla T4 chỉ có capability 7.5. Phương án thay thế là dùng GPU L4/A100 hoặc Colab trả phí để chạy FlashAttention/xformers tương thích. Tôi chọn giữ T4 miễn phí và chuyển sang PyTorch SDPA vì lỗi xuất phát từ giới hạn phần cứng của kernel, không phải lỗi dữ liệu preference hay công thức DPO.

Đổi sang SDPA làm quá trình train chậm hơn và tăng VRAM so với backend fused, nhưng đổi lại run hoàn thành đủ 250 step và tạo được adapter, reward-curve, side-by-side evaluation. Kết quả chỉ xác nhận một phần kỳ vọng: gap cuối train dương (+0.0900), nhưng cả chosen và rejected reward đều âm. Vì thế fix này giúp pipeline chạy được và tạo evidence cho rubric, nhưng không tự động bảo đảm chất lượng alignment cao. Nếu làm lại, tôi sẽ mount Google Drive ngay đầu notebook, sao lưu sau từng NB, đồng thời lưu metrics JSON và file adapter sau khi train xong. Tôi cũng sẽ chạy một beta-sweep nhỏ để xác định liệu gap dương hiện tại xuất phát từ cải thiện chosen hay chỉ vì rejected bị giảm mạnh hơn.

---

## 7. Benchmark interpretation

NB6 là bonus nên không chạy benchmark IFEval, GSM8K, MMLU và AlpacaEval-lite trong lần này. Do đó không nên suy diễn rằng kết quả thắng 4/8 trong manual rubric sẽ chuyển thành cải thiện trên mọi benchmark. Đánh giá NB4 chủ yếu đo các prompt Việt ngắn, gồm helpfulness và safety, còn GSM8K/MMLU đo năng lực suy luận và kiến thức theo cách khác. Một khả năng là DPO cải thiện cách từ chối unsafe prompt nhưng không tăng, thậm chí giảm nhẹ, điểm toán do alignment tax.

Nếu chạy NB6, tôi sẽ đối chiếu SFT-only và SFT+DPO trên cùng sampling seed, sau đó xem IFEval/AlpacaEval có nhất quán với win-rate NB4 không. Tôi cũng sẽ kiểm tra GSM8K và MMLU: nếu chúng giảm trong khi safety tăng thì đó là trade-off cần báo cáo thay vì gọi là cải thiện toàn diện. Với run hiện tại, kết luận giới hạn ở chỗ DPO có tín hiệu preference dương và kết quả định tính tốt hơn hoặc hoà trong cả 8 prompt; chưa có đủ dữ liệu để khẳng định generalization rộng.

---

## Bonus

- [ ] Đã làm beta-sweep
- [ ] Đã push HuggingFace Hub
- [ ] Đã release GGUF
- [ ] Đã chạy NB6 benchmark
- [x] Pair work theo quy định của lab

## Điều ngạc nhiên nhất

Một lỗi attention backend phụ thuộc compute capability có thể làm toàn bộ DPO training dừng dù model, dataset và trainer đều đúng. Đọc traceback đến dòng kernel yêu cầu SM 8.0 giúp phân biệt lỗi phần cứng với lỗi logic training.
