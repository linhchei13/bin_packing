# Tổng hợp các công thức và cách tính Lower Bound (LB) và Upper Bound (UB) cho bài toán 2D Bin Packing Problem (2D-BPP)

## 1. Giới thiệu

Bài toán 2D Bin Packing Problem (2D-BPP) là một bài toán tối ưu tổ hợp NP-hard, trong đó mục tiêu là đóng gói một tập hợp các vật phẩm hình chữ nhật có kích thước khác nhau vào số lượng thùng hình chữ nhật giống hệt nhau nhỏ nhất có thể. Bài toán này có nhiều ứng dụng thực tế trong các lĩnh vực như cắt vật liệu (gỗ, kim loại, kính), đóng gói sản phẩm, và sắp xếp kho bãi.

Để đánh giá chất lượng của các giải pháp heuristic hoặc xấp xỉ cho 2D-BPP, việc xác định các cận dưới (Lower Bound - LB) và cận trên (Upper Bound - UB) là rất quan trọng. LB cung cấp một giới hạn dưới cho số lượng thùng tối thiểu cần thiết, trong khi UB cung cấp một giới hạn trên (số lượng thùng mà một thuật toán cụ thể tìm được). Mục tiêu là tìm kiếm các giải pháp mà UB càng gần với LB càng tốt.

Phần này sẽ tổng hợp các công thức và phương pháp tính toán hiệu quả cho cả Lower Bound và Upper Bound trong bài toán 2D-BPP.




## 2. Lower Bound (LB)

Lower Bound (LB) là số lượng thùng tối thiểu cần thiết để đóng gói tất cả các vật phẩm. Vì 2D-BPP là bài toán NP-hard, việc tìm kiếm LB chặt nhất là rất khó. Tuy nhiên, có một số công thức và phương pháp để ước tính LB, cung cấp một giới hạn dưới cho giải pháp tối ưu.

### 2.1. Continuous Lower Bound (LBC)

Đây là công thức cơ bản nhất, dựa trên tổng diện tích của tất cả các vật phẩm và diện tích của một thùng. Công thức này không tính đến hình dạng của vật phẩm hay khả năng xoay của chúng. Nó chỉ đơn thuần là một ước tính dựa trên diện tích:

```
LBC = ceil(sum(wj * hj for j=1 to n) / (W * H))
```

Trong đó:
- `wj`: chiều rộng của vật phẩm `j`
- `hj`: chiều cao của vật phẩm `j`
- `W`: chiều rộng của thùng
- `H`: chiều cao của thùng
- `n`: tổng số vật phẩm
- `ceil()`: hàm làm tròn lên số nguyên gần nhất.

`LBC` là một cận dưới yếu vì nó bỏ qua các ràng buộc về hình học và sắp xếp. Tuy nhiên, nó dễ tính toán và thường được sử dụng làm điểm khởi đầu.

### 2.2. Lower Bound dựa trên việc phân tách vật phẩm thành hình vuông (Dell'Amico et al., 2002)

Bài báo "A lower bound for the non-oriented two-dimensional bin packing problem" của Dell'Amico, Martello và Vigo (2002) giới thiệu một phương pháp chặt chẽ hơn để tính Lower Bound, có tính đến khả năng xoay 90 độ của các vật phẩm. Ý tưởng chính là phân tách mỗi vật phẩm hình chữ nhật thành một tập hợp các vật phẩm hình vuông nhỏ hơn, sau đó áp dụng các nguyên tắc đóng gói cho các hình vuông này.

**Quy trình CUTSQ**: Mỗi vật phẩm `wj x hj` được cắt thành các hình vuông có cạnh `hj`. Phần còn lại (nếu có) sẽ được xoay 90 độ và tiếp tục quá trình cắt cho đến khi không thể cắt thêm hình vuông nào có kích thước lớn hơn 1.

Sau khi chuyển đổi tất cả các vật phẩm thành tập hợp các hình vuông `JSQ` với kích thước cạnh `lj`, các hình vuông này được phân loại thành các tập con dựa trên kích thước của chúng so với kích thước thùng `W` và `H`, và một tham số `q` (0 <= q <= H/2):

-   `S1 = {j ∈ JSQ : lj > W - q}`
-   `S2 = {j ∈ JSQ : W - q > lj > W/2}`
-   `S3 = {j ∈ JSQ : W/2 > lj > H/2}`
-   `S4 = {j ∈ JSQ : H/2 > lj > q}`

**Công thức Lower Bound (L(q))**: Dựa trên các tập con này, một công thức Lower Bound `L(q)` được định nghĩa:

```
L(q) = |S1| + L_tilde + max(0, ceil((sum(lj^2 for j in S2 U S3 U S4) - (W*H*L_tilde - sum(lj^2 for j in S23) - sum(lj*(H-lj) for j in (S2 U S3) \ S23))) / (W*H)))
```

Trong đó:
-   `L_tilde = |S2| + max(ceil(sum(lj for j in S3 \ SR3) / W), ceil(|S3 \ SR3| / (floor(H/2) + 1)))`
-   `SR3`: tập hợp các vật phẩm lớn nhất của `S3` có thể được đóng gói vào các thùng đã chứa `S2`.
-   `S23 = {j ∈ S2 U S3 : lj > H - q}`

**Lower Bound cuối cùng (LB)**: Để có được cận dưới chặt nhất từ phương pháp này, `LB` được lấy là giá trị lớn nhất của `L(q)` trên tất cả các giá trị `q` có thể:

```
LB = max {L(q)} for 0 <= q <= H/2
```

Phương pháp này cung cấp một cận dưới chặt chẽ hơn đáng kể so với `LBC` vì nó xem xét cấu trúc hình học và khả năng xoay của vật phẩm. Việc tính toán `LB` này có thể được thực hiện trong thời gian `O(m)` (cộng thêm `O(m log m)` để sắp xếp các vật phẩm), trong đó `m` là số lượng hình vuông được tạo ra từ quy trình `CUTSQ`.

**Trường hợp đặc biệt (đóng gói hình vuông vào hình vuông)**: Nếu `W = H` (tức là thùng là hình vuông), công thức `L(q)` đơn giản hóa thành:

```
L(q) = |S1 U S2| + max(0, ceil(sum(lj^2 for j in S2 U S4) / W^2) - |S2|)
```




## 3. Upper Bound (UB)

Upper Bound (UB) là số lượng thùng mà một thuật toán đóng gói cụ thể tìm được để chứa tất cả các vật phẩm. Vì 2D-BPP là bài toán NP-hard, việc tìm kiếm giải pháp tối ưu (số lượng thùng tối thiểu) là rất phức tạp. Do đó, các thuật toán heuristic và xấp xỉ thường được sử dụng để tìm kiếm các giải pháp gần tối ưu, và số lượng thùng mà các thuật toán này sử dụng chính là một Upper Bound.

Mục tiêu khi sử dụng các thuật toán này là để có được một UB càng gần với LB càng tốt, từ đó đánh giá hiệu quả của thuật toán.

### 3.1. Các phương pháp Heuristic phổ biến cung cấp Upper Bounds

Các thuật toán heuristic thường được phân loại dựa trên cách chúng sắp xếp và đặt các vật phẩm vào thùng. Dưới đây là một số phương pháp phổ biến:

#### 3.1.1. Thuật toán định hướng theo cấp độ (Level-oriented algorithms)

Các thuật toán này đóng gói các vật phẩm theo từng "cấp độ" hoặc "hàng" trong thùng. Vật phẩm được đặt từ trái sang phải trong một hàng cho đến khi không thể đặt thêm vật phẩm nào nữa hoặc không gian còn lại không đủ. Sau đó, một hàng mới sẽ được bắt đầu ở cấp độ tiếp theo.

-   **First Fit Decreasing Height (FFDH)**:
    1.  Sắp xếp tất cả các vật phẩm theo thứ tự giảm dần của chiều cao.
    2.  Tạo một hàng mới. Đặt vật phẩm đầu tiên vào hàng này.
    3.  Với mỗi vật phẩm còn lại, cố gắng đặt nó vào hàng hiện tại nếu nó vừa. Nếu không, tạo một hàng mới và đặt vật phẩm vào đó.
    4.  Lặp lại cho đến khi tất cả các vật phẩm được đặt.
    Số lượng hàng được sử dụng chính là Upper Bound.

-   **Next Fit Decreasing Height (NFDH)**:
    1.  Sắp xếp tất cả các vật phẩm theo thứ tự giảm dần của chiều cao.
    2.  Tạo một hàng mới. Đặt vật phẩm đầu tiên vào hàng này.
    3.  Với mỗi vật phẩm còn lại, cố gắng đặt nó vào hàng hiện tại. Nếu không vừa, **ngay lập tức** tạo một hàng mới và đặt vật phẩm vào đó (không quay lại các hàng trước đó).
    NFDH thường đơn giản hơn FFDH nhưng có thể cho UB kém chặt hơn.

#### 3.1.2. Heuristic điền từ dưới-trái (Bottom-Left-Fill - BLF)

Thuật toán BLF là một trong những heuristic phổ biến và hiệu quả cho 2D-BPP. Ý tưởng là đặt mỗi vật phẩm vào vị trí thấp nhất và sau đó là bên trái nhất có thể trong thùng hiện tại. Nếu không có vị trí nào trong thùng hiện tại, một thùng mới sẽ được mở.

-   **Cách hoạt động cơ bản của BLF**:
    1.  Chọn một vật phẩm (thường là vật phẩm lớn nhất hoặc theo một thứ tự ưu tiên nào đó).
    2.  Tìm kiếm vị trí trong thùng hiện tại bắt đầu từ góc dưới bên trái.
    3.  Đặt vật phẩm vào vị trí thấp nhất có thể mà không chồng lấn với các vật phẩm đã đặt.
    4.  Trong số các vị trí thấp nhất có thể, chọn vị trí bên trái nhất.
    5.  Nếu không tìm thấy vị trí trong thùng hiện tại, mở một thùng mới và lặp lại quá trình.

#### 3.1.3. Các thuật toán dựa trên Metaheuristics

Các phương pháp Metaheuristics không phải là công thức tính toán trực tiếp mà là các khung tìm kiếm giải pháp có thể được áp dụng cho 2D-BPP để tìm ra các UB rất chặt chẽ. Chúng thường tốn nhiều thời gian tính toán hơn nhưng có khả năng tìm ra các giải pháp gần tối ưu hơn.

-   **Genetic Algorithms (GA)**: Lấy cảm hứng từ quá trình chọn lọc tự nhiên, GA tạo ra một quần thể các giải pháp (cách sắp xếp vật phẩm), sau đó cải thiện chúng qua các thế hệ bằng cách sử dụng các toán tử như chọn lọc, lai ghép và đột biến.
-   **Simulated Annealing (SA)**: Mô phỏng quá trình ủ kim loại, SA khám phá không gian giải pháp bằng cách chấp nhận các thay đổi nhỏ, đôi khi là các thay đổi làm xấu đi giải pháp hiện tại để thoát khỏi các cực tiểu cục bộ.
-   **Tabu Search (TS)**: TS sử dụng một danh sách cấm (tabu list) để tránh lặp lại các giải pháp đã được khám phá gần đây, giúp tìm kiếm hiệu quả hơn trong không gian giải pháp.

### 3.2. Mối quan hệ giữa Lower Bound và Upper Bound

Trong nghiên cứu và thực hành, LB và UB được sử dụng để đánh giá chất lượng của các thuật toán. Khoảng cách giữa LB và UB càng nhỏ thì thuật toán càng hiệu quả. Nếu LB = UB, điều đó có nghĩa là thuật toán đã tìm thấy giải pháp tối ưu.

```
Chất lượng giải pháp = (UB - LB) / LB * 100%
```





## 4. Kết luận

Việc xác định Lower Bound và Upper Bound là rất quan trọng trong việc nghiên cứu và giải quyết bài toán 2D-BPP. Lower Bound cung cấp một giới hạn lý thuyết về số lượng thùng tối thiểu cần thiết, trong khi Upper Bound thể hiện hiệu suất thực tế của các thuật toán heuristic. Sự chênh lệch giữa hai giá trị này cho phép đánh giá chất lượng của các giải pháp được tìm thấy.

Các phương pháp tính Lower Bound ngày càng trở nên chặt chẽ hơn, đặc biệt là các phương pháp có tính đến cấu trúc hình học và khả năng xoay của vật phẩm. Đối với Upper Bound, các thuật toán heuristic và metaheuristic tiếp tục được phát triển để tìm ra các giải pháp ngày càng gần với tối ưu, mặc dù bài toán vẫn là một thách thức lớn trong lĩnh vực tối ưu tổ hợp.

Hy vọng tài liệu này cung cấp một cái nhìn tổng quan hữu ích về các công thức và phương pháp tính toán LB và UB cho bài toán 2D-BPP.


