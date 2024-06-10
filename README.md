# Table of Contents
# House Price Prediction | Machine Learning  Final Project
---


![](https://s3.eu-north-1.amazonaws.com/ammar-files/kaggle-kernels/House+Price+Prediction+%7C+An+End-to-End+Machine+Learning+Project/header-img.jpg)

<p>&nbsp;</p>

# Các Phần Chính

<a href="#introduction">Giới Thiệu</a>

<a href="#data-prep">Chuẩn bị dữ Liệu </a>

<a href="#eda">Exploratory Data Analysis</a>

<a href="#pred-type">Các Mô Hình Sử Dụng</a>

<a href="#model-building">Xây Dựng Và Đánh Giá Mô Hình </a>

<a href="#analysis-comparison">Phân Tích và So Sánh</a>

<a href="#comparison">Kết Luận</a>


---

## General Information
<h1 id="introduction">Giới thiệu</h1>

- Dự đoán giá nhà là một nhiệm vụ quan trọng và đầy thách thức trong lĩnh vực bất động sản và khoa học dữ liệu. Dự đoán chính xác về giá nhà có thể mang lại lợi ích cho nhiều bên liên quan, bao gồm người mua, người bán, nhà đầu tư và nhà hoạch định chính sách. Khả năng dự báo giá nhà với độ chính xác cao có thể giúp đưa ra quyết định sáng suốt hơn, lập kế hoạch tài chính tốt hơn và thị trường bất động sản hiệu quả hơn.

- Hàng ngàn ngôi nhà được bán mỗi ngày. Có một số câu hỏi mà người mua nào cũng tự hỏi mình như: Giá thực tế mà ngôi nhà này xứng đáng là bao nhiêu? Tôi có đang trả giá hợp lý không? Trong tập dữ liệu này, một mô hình học máy được đề xuất để dự đoán giá nhà dựa trên dữ liệu liên quan đến ngôi nhà (kích thước của nó, năm xây dựng, v.v.). 

Trong nghiên cứu này, chúng tôi sẽ sử dụng bộ dữ liệu về nhà ở do De Cock (2011) trình bày. Tập dữ liệu này mô tả doanh số bán nhà ở ở Ames, Iowa bắt đầu từ năm 2006 đến năm 2010. Tập dữ liệu chứa một số lượng lớn các biến liên quan đến việc xác định giá nhà. 

## Mô tả dữ liệu

Tập dữ liệu chứa các bản ghi `2930` (hàng) và các tính năng `82` (cột).
Ở đây, chúng tôi sẽ cung cấp một mô tả ngắn gọn về các tính năng của tập dữ liệu. Vì số lượng đặc điểm lớn (82), chúng tôi sẽ đính kèm tệp mô tả dữ liệu gốc vào bài viết này để biết thêm thông tin về tập dữ liệu 


|Feature|Description|
|-------|-----------|
|SalePrice|  giá bán tài sản bằng đô la. Đây là biến mục tiêu mà bạn đang cố gắng dự đoán.|
|MSSubClass|  Loại nhà liên quan đến việc bán|
|MSZoning| Phân loại phân vùng chung|
|LotFrontage| Feet tuyến tính của đường phố kết nối với bất động sản|
|LotArea|Kích thước lô tính bằng feet vuông|
|Street| Loại đường vào nhà |
|Alley| Loại đường vào ngõ|
|LotShape| Hình dạng chung của tài sản|
|LandContour| Độ bằng phẳng của bất động sản|
|Utilities| Loại tiện ích sẵn có|
|LotConfig| Cấu hình lô|
|LandSlope| Độ dốc của tài sản|
|Neighborhood|  Các vị trí  lân cận thực tế trong giới hạn thành phố Ames|
|Condition1| Gần đường chính hoặc đường sắt|
|Condition2| Gần đường chính hoặc đường sắt (nếu có giây)|
|BldgType| Loại nhà ở|
|HouseStyle| Phong cách nhà ở|
|OverallQual| Chất lượng vật liệu và hoàn thiện tổng thể|
|OverallCond| Đánh giá tình trạng tổng thể|
|YearBuilt| Năm Xây Dựng|
|YearRemodAdd| Năm sửa sang/thay đổi hoặc thêm|
|RoofStyle| Loại mái|
|RoofMatl| Vật liệu mái|
|Exterior1st| Lớp phủ bên ngoài ngôi nhà|
|Exterior2nd| Lớp phủ bên ngoài ngôi nhà (nếu có nhiều hơn một vật liệu)|
|MasVnrType| Loại veneer xây|
|MasVnrArea| Diện tích ván lạng tính bằng feet vuông|
|ExterQual| Chất lượng vật liệu bên ngoài|
|ExterCond| Tình trạng hiện tại của vật liệu ở bên ngoài|
|Foundation| Loại móng|
|BsmtQual| Chiều cao tầng hầm|
|BsmtCond| Hiện trạng chung tầng hầm|
|BsmtExposure| Tường tầng hầm lối đi hoặc sân vườn|
|BsmtFinType1| Chất lượng khu vực hoàn thiện tầng hầm|
|BsmtFinSF1| Loại 1 hoàn thiện feet vuông|
|BsmtFinType2|  Chất lượng của khu vực hoàn thiện thứ hai (nếu có)|
|BsmtFinSF2| Loại 2 hoàn thiện feet vuông|
|BsmtUnfSF| Diện tích tầng hầm chưa hoàn thiện|
|TotalBsmtSF| Tổng mét vuông diện tích tầng hầm|
|Heating| Loại hệ thống sưởi|
|HeatingQC| Chất lượng và tình trạng sưởi ấm|
|CentralAir| Điều hòa trung tâm|
|Electrical| Hệ thống điện|
|1stFlrSF| feet vuông tầng một|
|2ndFlrSF| feet vuông tầng hai|
|LowQualFinSF| Feet vuông hoàn thiện chất lượng thấp (tất cả các tầng)|
|GrLivArea| Diện tích sinh hoạt trên mặt đất (feet vuông)|
|BsmtFullBath| Phòng tắm đầy đủ ở tầng hầm|
|BsmtHalfBath| Phòng tắm ở tầng hầm|
|FullBath| Phòng tắm đầy đủ trên lớp|
|HalfBath| Tắm nửa trên lớp|
|Bedroom| Số phòng ngủ trên tầng hầm|
|Kitchen| Số lượng bếp|
|KitchenQual| Chất lượng nhà bếp|
|TotRmsAbvGrd| Tổng số phòng trên loại (không bao gồm phòng tắm)|
|Functional| Đánh giá chức năng của ngôi nhà|
|Fireplaces| Số lượng lò sưởi|
|FireplaceQu| Chất lượng lò sưởi|
|GarageType| vị trí gara|
|GarageYrBlt| Năm gara được xây dựng|
|GarageFinish| Hoàn thiện nội thất gara|
|GarageCars| Kích thước của gara tính theo sức chứa ô tô|
|GarageArea| Kích thước của gara tính bằng feet vuông|
|GarageQual| Chất lượng gara|
|GarageCond| Tình trạng gara|
|PavedDrive| Đường lái xe trải nhựa|
|WoodDeckSF| Diện tích sàn gỗ tính bằng feet vuông|
|OpenPorchSF| Diện tích hiên mở tính bằng feet vuông|
|EnclosedPorch|  Diện tích hiên bao quanh tính bằng feet vuông|
|3SsnPorch| Diện tích hiên nhà ba mùa tính bằng feet vuông|
|ScreenPorch| Màn hình diện tích hiên nhà tính bằng feet vuông|
|PoolArea| Diện tích hồ bơi tính bằng mét vuông|
|PoolQC| Chất lượng bể bơi|
|Fence| Chất lượng hàng rào|
|MiscFeature| Tính năng khác không được đề cập trong các danh mục khác|
|MiscVal| Giá trị $ của tính năng linh tinh|
|MoSold| Tháng bán được|
|YrSold| Năm bán được|
|SaleType| Loại hình bán hàng|
|SaleCondition| Tình trạng bán hàng|

---

## Problem Solving

### 👨‍🏫 Exploring the Dataset and Pre-processing
- Mô tả một cách tổng thể nhất để người đọc hiểu chính xác cấu trúc của tập dữ liệu này là gì.
- Sử dụng một số kỹ thuật trực quan dễ đọc để vạch ra các đặc điểm quan trọng của tập dữ liệu.
- Xác định bất kỳ điều gì bất thường trong tập dữ liệu, chẳng hạn như điểm dữ liệu null/nan hoặc các ngoại lệ, sẽ ảnh hưởng không chính xác đến quá trình phân tích.

### 📊 Set up prediction models using models such as Neutral network, Decision tree,...
- Bài toán này nhằm mục đích dự báo giá nhà bằng cách phân tích các thuộc tính đặc điểm có mối tương quan chặt chẽ với các biến Kết quả.
- Quan sát tập dữ liệu để xác định những thuộc tính nào không cần thiết cho những vấn đề này. Sau đó, chúng tôi sẽ loại bỏ chúng trước khi xây dựng mô hình học máy.
- So sánh hiệu năng, độ chính xác của đa dạng các mô hình và rút ra kết luận mô hình nào tốt hơn.

### 🗂 From the best Models come the best attributes
- Từ các mô hình dự đoán trên ta sẽ thấy được mô hình nào đưa ra được dự đoán tốt nhất dựa trên hệ số MSE ( càng nhỏ càng tốt) 
- Sau đó ta sẽ lấy các thuộc tính tốt nhất liên quan đến mô hình đó để đưa ra kết luận tốt hơn

---

## Technology
- **Environment**: Python 
- **Display mode**: Jupyter notebook
- **Model**:
  - `Liner Regression` for logistic regression.
  - `Elastic Net ` 
  - `KNN`.
  - `Support Vector Regression` for support vector.
  - `Decesion tree ` 
  - `Neural Network`
  - `Random Forest` 
  - `Gradient Boosting`

---

