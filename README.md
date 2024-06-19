# Đặc trưng âm thanh và vai trò trong phân loại thể loại âm nhạc

Các đặc trưng âm thanh được liệt kê dưới đây thường được sử dụng trong phân tích âm nhạc và nhận dạng giọng nói. Chúng cung cấp thông tin chi tiết về cấu trúc và đặc điểm của tín hiệu âm thanh, giúp cho việc phân loại các thể loại âm nhạc trở nên hiệu quả hơn.

## Danh sách các đặc trưng

- `chroma_stft_mean` và `chroma_stft_var`: Trung bình và độ biến thiên của chroma feature từ Short-Time Fourier Transform (STFT).
- `rms_mean` và `rms_var`: Giá trị trung bình và độ biến thiên của Root Mean Square (RMS), biểu diễn cường độ của âm thanh.
- `spectral_centroid_mean` và `spectral_centroid_var`: Trung bình và độ biến thiên của trọng tâm phổ, biểu diễn độ sáng của âm thanh.
- `spectral_bandwidth_mean` và `spectral_bandwidth_var`: Trung bình và độ biến thiên của băng thông phổ, đo lường độ rộng của phổ xung quanh trọng tâm phổ.
- `rolloff_mean` và `rolloff_var`: Trung bình và độ biến thiên của điểm rolloff phổ, biểu diễn tần số mà dưới đó một phần nhất định của tổng năng lượng phổ nằm.
- `zero_crossing_rate_mean` và `zero_crossing_rate_var`: Trung bình và độ biến thiên của tỷ lệ vượt qua không, đo lường số lần tín hiệu âm thanh vượt qua trục không trong một khoảng thời gian nhất định.
- `harmony_mean` và `harmony_var`: Trung bình và độ biến thiên của độ hài hòa, biểu diễn mức độ hòa hợp của các tần số trong tín hiệu âm thanh.
- `tempo`: Nhịp độ của bản nhạc hoặc tín hiệu âm thanh, thường được đo bằng số nhịp trên phút (BPM).
- `mfcc{i+1}_mean` và `mfcc{i+1}_var`: Trung bình và độ biến thiên của hệ số Mel-frequency cepstral thứ i, biểu diễn cấu trúc phổ của âm thanh trên thang Mel.

## Vai trò của các đặc trưng trong phân loại thể loại âm nhạc

- **Chroma STFT**: Phản ánh sự phân bố năng lượng theo các nốt nhạc, hữu ích để phân loại dựa trên hòa âm.
- **RMS**: Đo lường cường độ âm thanh, giúp phân biệt các thể loại nhạc dựa trên mức độ độc đáo của âm thanh.
- **Trọng tâm phổ và Băng thông phổ**: Cho biết độ sáng và độ rộng của âm thanh, hỗ trợ phân loại dựa trên cấu trúc phổ.
- **Điểm rolloff phổ và Tỷ lệ vượt qua không**: Biểu thị các đặc điểm độc đáo của âm thanh, giúp phân biệt các thể loại âm nhạc với các đặc trưng âm sắc khác nhau.
- **Độ hài hòa và Nhịp độ**: Thể hiện mức độ hòa hợp và sự đa dạng về tempo, quan trọng trong việc phân loại thể loại âm nhạc đa dạng.
- **MFCCs**: Biểu diễn cấu trúc phổ trên thang Mel, giúp phân biệt âm thanh và hình dung mẫu hình âm nhạc.

Các đặc trưng này khi kết hợp với nhau tạo thành một hệ thống mạnh mẽ để phân loại các thể loại âm nhạc khác nhau dựa trên các đặc điểm phổ biến và độc đáo của từng thể loại.

## Cách sử dụng Code
- Git clone 
- pip install -r requirements.txt
- Run file model/train_model.py nếu muốn thay train mô hình hoặc thay đổi tham số
- Run file app.py để chạy xem demo

