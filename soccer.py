from dataclasses import dataclass, field
from typing import List, Tuple

# TODO：配置自动读取

@dataclass
class SoccerPitchConfiguration:
    """标准足球场配置 (120m × 70m)"""
    width: int = 7000  # [cm] 70m
    length: int = 12000  # [cm] 120m
    penalty_box_width: int = 4100  # [cm]
    penalty_box_length: int = 2015  # [cm]
    goal_box_width: int = 1832  # [cm]
    goal_box_length: int = 550  # [cm]
    centre_circle_radius: int = 915  # [cm]
    penalty_spot_distance: int = 1100  # [cm]

    @property
    def vertices(self) -> List[Tuple[int, int]]:
        return [
            (0, 0),  # 1
            (0, (self.width - self.penalty_box_width) / 2),  # 2
            (0, (self.width - self.goal_box_width) / 2),  # 3
            (0, (self.width + self.goal_box_width) / 2),  # 4
            (0, (self.width + self.penalty_box_width) / 2),  # 5
            (0, self.width),  # 6
            (self.goal_box_length, (self.width - self.goal_box_width) / 2),  # 7
            (self.goal_box_length, (self.width + self.goal_box_width) / 2),  # 8
            (self.penalty_spot_distance, self.width / 2),  # 9
            (self.penalty_box_length, (self.width - self.penalty_box_width) / 2),  # 10
            (self.penalty_box_length, (self.width - self.goal_box_width) / 2),  # 11
            (self.penalty_box_length, (self.width + self.goal_box_width) / 2),  # 12
            (self.penalty_box_length, (self.width + self.penalty_box_width) / 2),  # 13
            (self.length / 2, 0),  # 14
            (self.length / 2, self.width / 2 - self.centre_circle_radius),  # 15
            (self.length / 2, self.width / 2 + self.centre_circle_radius),  # 16
            (self.length / 2, self.width),  # 17
            (
                self.length - self.penalty_box_length,
                (self.width - self.penalty_box_width) / 2
            ),  # 18
            (
                self.length - self.penalty_box_length,
                (self.width - self.goal_box_width) / 2
            ),  # 19
            (
                self.length - self.penalty_box_length,
                (self.width + self.goal_box_width) / 2
            ),  # 20
            (
                self.length - self.penalty_box_length,
                (self.width + self.penalty_box_width) / 2
            ),  # 21
            (self.length - self.penalty_spot_distance, self.width / 2),  # 22
            (
                self.length - self.goal_box_length,
                (self.width - self.goal_box_width) / 2
            ),  # 23
            (
                self.length - self.goal_box_length,
                (self.width + self.goal_box_width) / 2
            ),  # 24
            (self.length, 0),  # 25
            (self.length, (self.width - self.penalty_box_width) / 2),  # 26
            (self.length, (self.width - self.goal_box_width) / 2),  # 27
            (self.length, (self.width + self.goal_box_width) / 2),  # 28
            (self.length, (self.width + self.penalty_box_width) / 2),  # 29
            (self.length, self.width),  # 30
            (self.length / 2 - self.centre_circle_radius, self.width / 2),  # 31
            (self.length / 2 + self.centre_circle_radius, self.width / 2),  # 32
        ]

    edges: List[Tuple[int, int]] = field(default_factory=lambda: [
        (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (7, 8),
        (10, 11), (11, 12), (12, 13), (14, 15), (15, 16),
        (16, 17), (18, 19), (19, 20), (20, 21), (23, 24),
        (25, 26), (26, 27), (27, 28), (28, 29), (29, 30),
        (1, 14), (2, 10), (3, 7), (4, 8), (5, 13), (6, 17),
        (14, 25), (18, 26), (23, 27), (24, 28), (21, 29), (17, 30)
    ])

    labels: List[str] = field(default_factory=lambda: [
        "01", "02", "03", "04", "05", "06", "07", "08", "09", "10",
        "11", "12", "13", "15", "16", "17", "18", "20", "21", "22",
        "23", "24", "25", "26", "27", "28", "29", "30", "31", "32",
        "14", "19"
    ])

    colors: List[str] = field(default_factory=lambda: [
        "#FF1493", "#FF1493", "#FF1493", "#FF1493", "#FF1493", "#FF1493",
        "#FF1493", "#FF1493", "#FF1493", "#FF1493", "#FF1493", "#FF1493",
        "#FF1493", "#00BFFF", "#00BFFF", "#00BFFF", "#00BFFF", "#FF6347",
        "#FF6347", "#FF6347", "#FF6347", "#FF6347", "#FF6347", "#FF6347",
        "#FF6347", "#FF6347", "#FF6347", "#FF6347", "#FF6347", "#FF6347",
        "#00BFFF", "#00BFFF"
    ])

@dataclass
class HalfPitchConfiguration:
    """标准足球半场配置 (60m × 70m)
    
    特点：包含底线中点、小禁区长边中点、大禁区长边中点
    """
    width: int = 7000  # [cm] 70m
    length: int = 6000  # [cm] 60m (全场120m的一半)
    penalty_box_width: int = 4100  # [cm]
    penalty_box_length: int = 2015  # [cm]
    goal_box_width: int = 1832  # [cm]
    goal_box_length: int = 550  # [cm]
    centre_circle_radius: int = 915  # [cm]
    penalty_spot_distance: int = 1100  # [cm]

    @property
    def vertices(self) -> List[Tuple[int, int]]:
        return [
            # --- 原有顶点 1-19 ---
            (0, 0),  # 1. 左上角
            (0, (self.width - self.penalty_box_width) / 2),  # 2. 左底线-大禁区上
            (0, (self.width - self.goal_box_width) / 2),  # 3. 左底线-小禁区上
            (0, (self.width + self.goal_box_width) / 2),  # 4. 左底线-小禁区下
            (0, (self.width + self.penalty_box_width) / 2),  # 5. 左底线-大禁区下
            (0, self.width),  # 6. 左下角
            
            (self.goal_box_length, (self.width - self.goal_box_width) / 2),  # 7. 小禁区右上
            (self.goal_box_length, (self.width + self.goal_box_width) / 2),  # 8. 小禁区右下
            
            (self.penalty_spot_distance, self.width / 2),  # 9. 点球点
            
            (self.penalty_box_length, (self.width - self.penalty_box_width) / 2),  # 10. 大禁区右上
            (self.penalty_box_length, (self.width - self.goal_box_width) / 2),  # 11. 大禁区线-对应小禁区上
            (self.penalty_box_length, (self.width + self.goal_box_width) / 2),  # 12. 大禁区线-对应小禁区下
            (self.penalty_box_length, (self.width + self.penalty_box_width) / 2),  # 13. 大禁区右下
            
            (self.length, 0),  # 14. 中线上端
            (self.length, self.width / 2 - self.centre_circle_radius),  # 15. 中圈上端
            (self.length, self.width / 2 + self.centre_circle_radius),  # 16. 中圈下端
            (self.length, self.width),  # 17. 中线下端
            
            (self.length - self.centre_circle_radius, self.width / 2),  # 18. 中圈最左点
            (self.length, self.width / 2),  # 19. 中点 (开球点)

            # --- 新增中轴线关键点 (追加在末尾以保持Edges索引正确) ---
            (0, self.width / 2),  # 20. 【新增】3和4的中点 (球门线中点)
            (self.goal_box_length, self.width / 2),  # 21. 【新增】7和8的中点 (小禁区前沿中点)
            (self.penalty_box_length, self.width / 2),  # 22. 【新增】11和12的中点 (大禁区前沿中点)
        ]

    # 定义连线 (索引从1开始，保持不变)
    edges: List[Tuple[int, int]] = field(default_factory=lambda: [
        # 左底线 segments
        (1, 2), (2, 3), (3, 4), (4, 5), (5, 6),
        
        # 小禁区
        (3, 7), (7, 8), (8, 4),
        
        # 大禁区
        (2, 10), (10, 11), (11, 12), (12, 13), (13, 5),
        
        # 上下边线
        (1, 14), (6, 17),
        
        # 中线 segments (穿过中点)
        (14, 15), (15, 19), (19, 16), (16, 17)
    ])

    labels: List[str] = field(default_factory=lambda: [
        "01", "02", "03", "04", "05", "06", 
        "07", "08", "09", 
        "10", "11", "12", "13", 
        "14", "15", "16", "17", 
        "18", "19",
        "20", "21", "22"  # 新增标签
    ])

    colors: List[str] = field(default_factory=lambda: [
        "#FF1493", "#FF1493", "#FF1493", "#FF1493", "#FF1493", "#FF1493", # 底线区域
        "#FF1493", "#FF1493", # 小禁区
        "#FF1493", # 点球点
        "#FF1493", "#FF1493", "#FF1493", "#FF1493", # 大禁区
        "#00BFFF", "#00BFFF", "#00BFFF", "#00BFFF", # 中线
        "#00BFFF", "#00BFFF", # 中圈
        "#FFFFFF", "#FFFFFF", "#FFFFFF" # 新增点的颜色 (白色，区别于其他点)
    ])

@dataclass
class QingYuanConfiguration:
    width: int = 3600  # [cm]
    length: int = 5800  # [cm]
    penalty_box_width: int = 2490  # [cm]
    penalty_box_length: int = 990  # [cm]
    goal_box_width: int = 1440  # [cm]
    goal_box_length: int = 380  # [cm]
    centre_circle_radius: int = 585  # [cm]
    penalty_spot_distance: int = 730  # [cm]

    @property
    def vertices(self) -> List[Tuple[int, int]]:
        # 基础点 (复制自 SoccerPitchConfiguration, ID 1-32)
        base_vertices = [
            (0, 0),  # 1
            (0, (self.width - self.penalty_box_width) / 2),  # 2
            (0, (self.width - self.goal_box_width) / 2),  # 3
            (0, (self.width + self.goal_box_width) / 2),  # 4
            (0, (self.width + self.penalty_box_width) / 2),  # 5
            (0, self.width),  # 6
            (self.goal_box_length, (self.width - self.goal_box_width) / 2),  # 7
            (self.goal_box_length, (self.width + self.goal_box_width) / 2),  # 8
            (self.penalty_spot_distance, self.width / 2),  # 9
            (self.penalty_box_length, (self.width - self.penalty_box_width) / 2),  # 10
            (self.penalty_box_length, (self.width - self.goal_box_width) / 2),  # 11
            (self.penalty_box_length, (self.width + self.goal_box_width) / 2),  # 12
            (self.penalty_box_length, (self.width + self.penalty_box_width) / 2),  # 13
            (self.length / 2, 0),  # 14 (中线上)
            (self.length / 2, self.width / 2 - self.centre_circle_radius),  # 15
            (self.length / 2, self.width / 2 + self.centre_circle_radius),  # 16
            (self.length / 2, self.width),  # 17 (中线下)
            (self.length - self.penalty_box_length, (self.width - self.penalty_box_width) / 2),  # 18
            (self.length - self.penalty_box_length, (self.width - self.goal_box_width) / 2),  # 19
            (self.length - self.penalty_box_length, (self.width + self.goal_box_width) / 2),  # 20
            (self.length - self.penalty_box_length, (self.width + self.penalty_box_width) / 2),  # 21
            (self.length - self.penalty_spot_distance, self.width / 2),  # 22
            (self.length - self.goal_box_length, (self.width - self.goal_box_width) / 2),  # 23
            (self.length - self.goal_box_length, (self.width + self.goal_box_width) / 2),  # 24
            (self.length, 0),  # 25
            (self.length, (self.width - self.penalty_box_width) / 2),  # 26
            (self.length, (self.width - self.goal_box_width) / 2),  # 27
            (self.length, (self.width + self.goal_box_width) / 2),  # 28
            (self.length, (self.width + self.penalty_box_width) / 2),  # 29
            (self.length, self.width),  # 30
            (self.length / 2 - self.centre_circle_radius, self.width / 2),  # 31
            (self.length / 2 + self.centre_circle_radius, self.width / 2),  # 32
        ]
        
        # 新增：草皮分界点 (15等分)
        # 上边线 (y=0) 和 下边线 (y=width)
        segments = 15
        step = self.length / segments
        
        # 追加点
        extra_vertices = []
        for i in range(1, segments):
            x = i * step
            # 如果 x 接近中线 (length/2)，是否跳过？
            # 5800/15 = 386.66; 中线在 2900
            # 2900 / 386.66 = 7.5
            # 所以分点不会重合，无需跳过。
            
            extra_vertices.append((x, 0))          # 上边线分点
            extra_vertices.append((x, self.width)) # 下边线分点
            
        return base_vertices + extra_vertices

    # 基础边线 + 新增的边线关系
    # 注意：为了让工具能画出线，我们需要手动定义边线连接，但对于点标定来说，edges 主要是可视化辅助。
    # 这里保持基础 edges 即可，或者增加新点之间的连接（太繁琐，暂时保留基础架构）
    edges: List[Tuple[int, int]] = field(default_factory=lambda: [
        (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (7, 8),
        (10, 11), (11, 12), (12, 13), (14, 15), (15, 16),
        (16, 17), (18, 19), (19, 20), (20, 21), (23, 24),
        (25, 26), (26, 27), (27, 28), (28, 29), (29, 30),
        (1, 14), (2, 10), (3, 7), (4, 8), (5, 13), (6, 17),
        (14, 25), (18, 26), (23, 27), (24, 28), (21, 29), (17, 30)
    ])

    @property
    def labels(self) -> List[str]:
        # 基础标签 (32个)
        base_labels = [
            "01", "02", "03", "04", "05", "06", "07", "08", "09", "10",
            "11", "12", "13", "15", "16", "17", "18", "20", "21", "22",
            "23", "24", "25", "26", "27", "28", "29", "30", "31", "32",
            "14", "19"
        ]
        # 新增标签 (28个: 14上 + 14下)
        extra_labels = []
        for i in range(1, 15):
            extra_labels.append(f"T{i}") # Top
            extra_labels.append(f"B{i}") # Bottom
        return base_labels + extra_labels

    @property
    def colors(self) -> List[str]:
        # 基础颜色 (32个)
        base_colors = [
            "#FF1493", "#FF1493", "#FF1493", "#FF1493", "#FF1493", "#FF1493",
            "#FF1493", "#FF1493", "#FF1493", "#FF1493", "#FF1493", "#FF1493",
            "#FF1493", "#00BFFF", "#00BFFF", "#00BFFF", "#00BFFF", "#FF6347",
            "#FF6347", "#FF6347", "#FF6347", "#FF6347", "#FF6347", "#FF6347",
            "#FF6347", "#FF6347", "#FF6347", "#FF6347", "#FF6347", "#FF6347",
            "#00BFFF", "#00BFFF"
        ]
        # 新增颜色 (使用绿色区分)
        extra_colors = ["#00FF00"] * 28
        return base_colors + extra_colors

@dataclass
class HalfQingYuanConfiguration:
    width: int = 3600  # [cm]
    length: int = 2900  # [cm]
    penalty_box_width: int = 2490  # [cm]
    penalty_box_length: int = 990  # [cm]
    goal_box_width: int = 1440  # [cm]
    goal_box_length: int = 380  # [cm]
    centre_circle_radius: int = 585  # [cm]
    penalty_spot_distance: int = 730  # [cm]

    @property
    def vertices(self) -> List[Tuple[int, int]]:
        # 基础点 (复制自 HalfPitchConfiguration, ID 1-22)
        base_vertices = [
            (0, 0),  # 1
            (0, (self.width - self.penalty_box_width) / 2),  # 2
            (0, (self.width - self.goal_box_width) / 2),  # 3
            (0, (self.width + self.goal_box_width) / 2),  # 4
            (0, (self.width + self.penalty_box_width) / 2),  # 5
            (0, self.width),  # 6
            (self.goal_box_length, (self.width - self.goal_box_width) / 2),  # 7
            (self.goal_box_length, (self.width + self.goal_box_width) / 2),  # 8
            (self.penalty_spot_distance, self.width / 2),  # 9
            (self.penalty_box_length, (self.width - self.penalty_box_width) / 2),  # 10
            (self.penalty_box_length, (self.width - self.goal_box_width) / 2),  # 11
            (self.penalty_box_length, (self.width + self.goal_box_width) / 2),  # 12
            (self.penalty_box_length, (self.width + self.penalty_box_width) / 2),  # 13
            (self.length, 0),  # 14
            (self.length, self.width / 2 - self.centre_circle_radius),  # 15
            (self.length, self.width / 2 + self.centre_circle_radius),  # 16
            (self.length, self.width),  # 17
            (self.length - self.centre_circle_radius, self.width / 2),  # 18
            (self.length, self.width / 2),  # 19
            (0, self.width / 2),  # 20
            (self.goal_box_length, self.width / 2),  # 21
            (self.penalty_box_length, self.width / 2),  # 22
        ]
        
        # 新增：半场草皮分界点 (7等分)
        # 全场是 15 等分，半场就是 7.5 等分。
        # 分段长度 Step 保持与全场一致：FullLength (5800) / 15
        full_length = 5800
        step = full_length / 15
        
        # 半场范围内有 7 个完整分点 (i=1..7)
        extra_vertices = []
        for i in range(1, 8):
            x = i * step
            extra_vertices.append((x, 0))          # 上边线分点
            extra_vertices.append((x, self.width)) # 下边线分点
            
        return base_vertices + extra_vertices

    # 基础 edges
    edges: List[Tuple[int, int]] = field(default_factory=lambda: [
        (1, 2), (2, 3), (3, 4), (4, 5), (5, 6),
        (3, 7), (7, 8), (8, 4),
        (2, 10), (10, 11), (11, 12), (12, 13), (13, 5),
        (1, 14), (6, 17),
        (14, 15), (15, 19), (19, 16), (16, 17)
    ])

    @property
    def labels(self) -> List[str]:
        base_labels = [
            "01", "02", "03", "04", "05", "06", 
            "07", "08", "09", 
            "10", "11", "12", "13", 
            "14", "15", "16", "17", 
            "18", "19",
            "20", "21", "22"
        ]
        # 新增标签 (14个: 7上 + 7下)
        extra_labels = []
        for i in range(1, 8):
            extra_labels.append(f"T{i}") # Top
            extra_labels.append(f"B{i}") # Bottom
        return base_labels + extra_labels

    @property
    def colors(self) -> List[str]:
        base_colors = [
            "#FF1493", "#FF1493", "#FF1493", "#FF1493", "#FF1493", "#FF1493",
            "#FF1493", "#FF1493", "#FF1493", "#FF1493", "#FF1493", "#FF1493",
            "#FF1493", "#00BFFF", "#00BFFF", "#00BFFF", "#00BFFF", "#00BFFF",
            "#00BFFF", "#FFFFFF", "#FFFFFF", "#FFFFFF"
        ]
        # 新增颜色
        extra_colors = ["#00FF00"] * 14
        return base_colors + extra_colors

@dataclass
class WuChuanConfiguration:
    """吴川球场配置 (66m × 44m)"""
    width: int = 4400  # [cm] 44m
    length: int = 6600  # [cm] 66m
    penalty_box_width: int = 2400  # [cm] 大禁区宽24m (平行于底线)
    penalty_box_length: int = 1000  # [cm] 大禁区长10m (从底线延伸)
    goal_box_width: int = 1400  # [cm] 小禁区宽14m (平行于底线)
    goal_box_length: int = 400  # [cm] 小禁区长4m (从底线延伸)
    centre_circle_radius: int = 550  # [cm] 中圈半径5.5m
    penalty_spot_distance: int = 700  # [cm] 点球点到底线7m

    @property
    def vertices(self) -> List[Tuple[int, int]]:
        # 基础点 (1-32)
        base_vertices = [
            (0, 0),  # 1. 左上角
            (0, (self.width - self.penalty_box_width) / 2),  # 2. 左底线-大禁区上
            (0, (self.width - self.goal_box_width) / 2),  # 3. 左底线-小禁区上
            (0, (self.width + self.goal_box_width) / 2),  # 4. 左底线-小禁区下
            (0, (self.width + self.penalty_box_width) / 2),  # 5. 左底线-大禁区下
            (0, self.width),  # 6. 左下角
            
            (self.goal_box_length, (self.width - self.goal_box_width) / 2),  # 7. 小禁区右上
            (self.goal_box_length, (self.width + self.goal_box_width) / 2),  # 8. 小禁区右下
            
            (self.penalty_spot_distance, self.width / 2),  # 9. 点球点
            
            (self.penalty_box_length, (self.width - self.penalty_box_width) / 2),  # 10. 大禁区右上
            (self.penalty_box_length, (self.width - self.goal_box_width) / 2),  # 11. 大禁区线-对应小禁区上
            (self.penalty_box_length, (self.width + self.goal_box_width) / 2),  # 12. 大禁区线-对应小禁区下
            (self.penalty_box_length, (self.width + self.penalty_box_width) / 2),  # 13. 大禁区右下
            
            (self.length / 2, 0),  # 14. 中线上端
            (self.length / 2, self.width / 2 - self.centre_circle_radius),  # 15. 中圈上端
            (self.length / 2, self.width / 2 + self.centre_circle_radius),  # 16. 中圈下端
            (self.length / 2, self.width),  # 17. 中线下端
            
            (self.length - self.penalty_box_length, (self.width - self.penalty_box_width) / 2),  # 18. 右大禁区上
            (self.length - self.penalty_box_length, (self.width - self.goal_box_width) / 2),  # 19. 右大禁区线-对应小禁区上
            (self.length - self.penalty_box_length, (self.width + self.goal_box_width) / 2),  # 20. 右大禁区线-对应小禁区下
            (self.length - self.penalty_box_length, (self.width + self.penalty_box_width) / 2),  # 21. 右大禁区下
            
            (self.length - self.penalty_spot_distance, self.width / 2),  # 22. 右点球点
            
            (self.length - self.goal_box_length, (self.width - self.goal_box_width) / 2),  # 23. 右小禁区左上
            (self.length - self.goal_box_length, (self.width + self.goal_box_width) / 2),  # 24. 右小禁区左下
            
            (self.length, 0),  # 25. 右上角
            (self.length, (self.width - self.penalty_box_width) / 2),  # 26. 右底线-大禁区上
            (self.length, (self.width - self.goal_box_width) / 2),  # 27. 右底线-小禁区上
            (self.length, (self.width + self.goal_box_width) / 2),  # 28. 右底线-小禁区下
            (self.length, (self.width + self.penalty_box_width) / 2),  # 29. 右底线-大禁区下
            (self.length, self.width),  # 30. 右下角
            
            (self.length / 2 - self.centre_circle_radius, self.width / 2),  # 31. 中圈左点
            (self.length / 2 + self.centre_circle_radius, self.width / 2),  # 32. 中圈右点
        ]
        
        # 必需关键点 (33-38)
        required_points = [
            (0, self.width / 2),  # 33. 左底线中点 ⭐
            (self.goal_box_length, self.width / 2),  # 34. 小禁区长边中点(左) ⭐
            (self.penalty_box_length, self.width / 2),  # 35. 大禁区长边中点(左) ⭐
            (self.length - self.penalty_box_length, self.width / 2),  # 36. 大禁区长边中点(右) ⭐
            (self.length - self.goal_box_length, self.width / 2),  # 37. 小禁区长边中点(右) ⭐
            (self.length, self.width / 2),  # 38. 右底线中点 ⭐
        ]
        
        # 草皮分界点 (21等分长边)
        segments = 21
        step = self.length / segments
        
        extra_vertices = []
        for i in range(1, segments):
            x = i * step
            extra_vertices.append((x, 0))          # 上边线分点
            extra_vertices.append((x, self.width)) # 下边线分点
        
        return base_vertices + required_points + extra_vertices

    edges: List[Tuple[int, int]] = field(default_factory=lambda: [
        (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (7, 8),
        (10, 11), (11, 12), (12, 13), (14, 15), (15, 16),
        (16, 17), (18, 19), (19, 20), (20, 21), (23, 24),
        (25, 26), (26, 27), (27, 28), (28, 29), (29, 30),
        (1, 14), (2, 10), (3, 7), (4, 8), (5, 13), (6, 17),
        (14, 25), (18, 26), (23, 27), (24, 28), (21, 29), (17, 30)
    ])

    @property
    def labels(self) -> List[str]:
        # 基础标签 (1-32)
        base_labels = [
            "01", "02", "03", "04", "05", "06", "07", "08", "09", "10",
            "11", "12", "13", "15", "16", "17", "18", "20", "21", "22",
            "23", "24", "25", "26", "27", "28", "29", "30", "31", "32",
            "14", "19"
        ]
        
        # 必需关键点标签 (33-38)
        required_labels = ["33", "34", "35", "36", "37", "38"]
        
        # 草皮分界点标签 (20上 + 20下 = 40个)
        extra_labels = []
        for i in range(1, 21):
            extra_labels.append(f"T{i}")  # Top
            extra_labels.append(f"B{i}")  # Bottom
        
        return base_labels + required_labels + extra_labels

    @property
    def colors(self) -> List[str]:
        # 基础颜色 (32个)
        base_colors = [
            "#FF1493", "#FF1493", "#FF1493", "#FF1493", "#FF1493", "#FF1493",
            "#FF1493", "#FF1493", "#FF1493", "#FF1493", "#FF1493", "#FF1493",
            "#FF1493", "#00BFFF", "#00BFFF", "#00BFFF", "#00BFFF", "#FF6347",
            "#FF6347", "#FF6347", "#FF6347", "#FF6347", "#FF6347", "#FF6347",
            "#FF6347", "#FF6347", "#FF6347", "#FF6347", "#FF6347", "#FF6347",
            "#00BFFF", "#00BFFF"
        ]
        
        # 必需关键点颜色 (6个 - 白色突出显示)
        required_colors = ["#FFFFFF"] * 6
        
        # 草皮分界点颜色 (40个 - 绿色)
        extra_colors = ["#00FF00"] * 40
        
        return base_colors + required_colors + extra_colors


@dataclass
class WuChuanYiZhongConfiguration(WuChuanConfiguration):
    """吴川一中全场配置（68m × 46.5m）。

    吴川一中的场地标线尺寸与通用吴川场地不同，因此单独保留一个配置类，
    避免修改已有吴川项目的标定坐标。
    """

    width: int = 4650  # [cm] 场地宽 46.5m
    length: int = 6800  # [cm] 场地长 68m
    penalty_box_width: int = 2550  # [cm] 大禁区横向宽度 25.5m
    penalty_box_length: int = 995  # [cm] 大禁区向场内深度 9.95m
    goal_box_width: int = 1350  # [cm] 小禁区横向宽度 13.5m
    goal_box_length: int = 400  # [cm] 小禁区向场内深度 4m
    centre_circle_radius: int = 600  # [cm] 中圈半径 6m
    penalty_spot_distance: int = 900  # [cm] 点球点到底线 9m


@dataclass
class GaoWuConfiguration(SoccerPitchConfiguration):
    """高屋小学全场配置（48.92m × 32.24m）。"""

    width: int = 3224  # [cm] 32.24m
    length: int = 4892  # [cm] 48.92m
    penalty_box_width: int = 1770  # [cm] 大禁区横向宽度（平行于底线）17.7m
    penalty_box_length: int = 684  # [cm] 大禁区向场内深度 6.84m
    goal_box_width: int = 948  # [cm] 小禁区横向宽度 9.48m
    goal_box_length: int = 278  # [cm] 小禁区向场内深度 2.78m
    centre_circle_radius: int = 417  # [cm] 中圈半径 4.17m
    penalty_spot_distance: int = 630  # [cm] 点球点到底线 6.3m

    @property
    def vertices(self) -> List[Tuple[int, int]]:
        # 复用全场基础关键点，并补充上下边线的草皮分界点。
        base_vertices = super().vertices
        segments = 15
        step = self.length / segments
        extra_vertices = []
        for i in range(1, segments):
            x = i * step
            extra_vertices.append((x, 0))
            extra_vertices.append((x, self.width))
        return base_vertices + extra_vertices

    labels: List[str] = field(default_factory=lambda: (
        SoccerPitchConfiguration().labels + [
            label
            for i in range(1, 15)
            for label in (f"T{i}", f"B{i}")
        ]
    ))

    colors: List[str] = field(default_factory=lambda: (
        SoccerPitchConfiguration().colors + ["#00FF00"] * 28
    ))


@dataclass
class HalfGaoWuConfiguration(HalfPitchConfiguration):
    """高屋小学半场配置（24.46m × 32.24m）。"""

    width: int = 3224  # [cm] 32.24m
    length: int = 2446  # [cm] 24.46m
    penalty_box_width: int = 1770  # [cm] 大禁区横向宽度（平行于底线）17.7m
    penalty_box_length: int = 684  # [cm] 大禁区向场内深度 6.84m
    goal_box_width: int = 948  # [cm] 小禁区横向宽度 9.48m
    goal_box_length: int = 278  # [cm] 小禁区向场内深度 2.78m
    centre_circle_radius: int = 417  # [cm] 中圈半径 4.17m
    penalty_spot_distance: int = 630  # [cm] 点球点到底线 6.3m

    @property
    def vertices(self) -> List[Tuple[int, int]]:
        # 与清源小学半场一致，按全场 15 等分补充半场内的 7 组上下边线点。
        base_vertices = super().vertices
        full_length = self.length * 2
        step = full_length / 15
        extra_vertices = []
        for i in range(1, 8):
            x = i * step
            extra_vertices.append((x, 0))
            extra_vertices.append((x, self.width))
        return base_vertices + extra_vertices

    labels: List[str] = field(default_factory=lambda: (
        HalfPitchConfiguration().labels + [
            label
            for i in range(1, 8)
            for label in (f"T{i}", f"B{i}")
        ]
    ))

    colors: List[str] = field(default_factory=lambda: (
        HalfPitchConfiguration().colors + ["#00FF00"] * 14
    ))


@dataclass
class EliteU12Configuration(GaoWuConfiguration):
    """精英队 U12 全场配置（95m × 60m）。"""

    width: int = 6000  # [cm] 场地宽 60m
    length: int = 9500  # [cm] 场地长 95m
    penalty_box_width: int = 4000  # [cm] 大禁区横向宽度 40m
    penalty_box_length: int = 1700  # [cm] 大禁区向场内深度 17m
    goal_box_width: int = 1900  # [cm] 小禁区横向宽度 19m
    goal_box_length: int = 600  # [cm] 小禁区向场内深度 6m
    centre_circle_radius: int = 900  # [cm] 中圈半径 9m
    penalty_spot_distance: int = 1100  # [cm] 点球点到底线 11m


@dataclass
class HalfEliteU12Configuration(HalfGaoWuConfiguration):
    """精英队 U12 半场配置（47.5m × 60m）。"""

    width: int = 6000  # [cm] 场地宽 60m
    length: int = 4750  # [cm] 半场长 47.5m
    penalty_box_width: int = 4000  # [cm] 大禁区横向宽度 40m
    penalty_box_length: int = 1700  # [cm] 大禁区向场内深度 17m
    goal_box_width: int = 1900  # [cm] 小禁区横向宽度 19m
    goal_box_length: int = 600  # [cm] 小禁区向场内深度 6m
    centre_circle_radius: int = 900  # [cm] 中圈半径 9m
    penalty_spot_distance: int = 1100  # [cm] 点球点到底线 11m


@dataclass
class EliteU15Configuration(GaoWuConfiguration):
    """精英队 U15 全场配置（95m × 60m）。"""

    width: int = 6000  # [cm] 场地宽 60m
    length: int = 9500  # [cm] 场地长 95m
    penalty_box_width: int = 4000  # [cm] 大禁区横向宽度 40m
    penalty_box_length: int = 1700  # [cm] 大禁区向场内深度 17m
    goal_box_width: int = 1900  # [cm] 小禁区横向宽度 19m
    goal_box_length: int = 600  # [cm] 小禁区向场内深度 6m
    centre_circle_radius: int = 900  # [cm] 中圈半径 9m
    penalty_spot_distance: int = 1100  # [cm] 点球点到底线 11m


@dataclass
class HalfEliteU15Configuration(HalfGaoWuConfiguration):
    """精英队 U15 半场配置（47.5m × 60m）。"""

    width: int = 6000  # [cm] 场地宽 60m
    length: int = 4750  # [cm] 半场长 47.5m
    penalty_box_width: int = 4000  # [cm] 大禁区横向宽度 40m
    penalty_box_length: int = 1700  # [cm] 大禁区向场内深度 17m
    goal_box_width: int = 1900  # [cm] 小禁区横向宽度 19m
    goal_box_length: int = 600  # [cm] 小禁区向场内深度 6m
    centre_circle_radius: int = 900  # [cm] 中圈半径 9m
    penalty_spot_distance: int = 1100  # [cm] 点球点到底线 11m


@dataclass
class PuGongYingU11Configuration(EliteU12Configuration):
    """蒲公英 U11 全场配置，与精英队 U12 共用场地参数。"""


@dataclass
class HalfPuGongYingU11Configuration(HalfEliteU12Configuration):
    """蒲公英 U11 半场配置，与精英队 U12 共用场地参数。"""


@dataclass
class HalfWuChuanConfiguration:
    """吴川半场配置 (33m × 44m)"""
    width: int = 4400  # [cm] 44m
    length: int = 3300  # [cm] 33m (半场)
    penalty_box_width: int = 2400  # [cm] 大禁区宽24m (平行于底线)
    penalty_box_length: int = 1000  # [cm] 大禁区长10m (从底线延伸)
    goal_box_width: int = 1400  # [cm] 小禁区宽14m (平行于底线)
    goal_box_length: int = 400  # [cm] 小禁区长4m (从底线延伸)
    centre_circle_radius: int = 550  # [cm] 中圈半径5.5m
    penalty_spot_distance: int = 700  # [cm] 点球点到底线7m

    @property
    def vertices(self) -> List[Tuple[int, int]]:
        # 基础点 (1-19)
        base_vertices = [
            (0, 0),  # 1. 左上角
            (0, (self.width - self.penalty_box_width) / 2),  # 2. 左底线-大禁区上
            (0, (self.width - self.goal_box_width) / 2),  # 3. 左底线-小禁区上
            (0, (self.width + self.goal_box_width) / 2),  # 4. 左底线-小禁区下
            (0, (self.width + self.penalty_box_width) / 2),  # 5. 左底线-大禁区下
            (0, self.width),  # 6. 左下角
            
            (self.goal_box_length, (self.width - self.goal_box_width) / 2),  # 7. 小禁区右上
            (self.goal_box_length, (self.width + self.goal_box_width) / 2),  # 8. 小禁区右下
            
            (self.penalty_spot_distance, self.width / 2),  # 9. 点球点
            
            (self.penalty_box_length, (self.width - self.penalty_box_width) / 2),  # 10. 大禁区右上
            (self.penalty_box_length, (self.width - self.goal_box_width) / 2),  # 11. 大禁区线-对应小禁区上
            (self.penalty_box_length, (self.width + self.goal_box_width) / 2),  # 12. 大禁区线-对应小禁区下
            (self.penalty_box_length, (self.width + self.penalty_box_width) / 2),  # 13. 大禁区右下
            
            (self.length, 0),  # 14. 中线上端
            (self.length, self.width / 2 - self.centre_circle_radius),  # 15. 中圈上端
            (self.length, self.width / 2 + self.centre_circle_radius),  # 16. 中圈下端
            (self.length, self.width),  # 17. 中线下端
            
            (self.length - self.centre_circle_radius, self.width / 2),  # 18. 中圈左点
            (self.length, self.width / 2),  # 19. 中点
        ]
        
        # 必需关键点 (20-22)
        required_points = [
            (0, self.width / 2),  # 20. 底线中点 ⭐
            (self.goal_box_length, self.width / 2),  # 21. 小禁区长边中点 ⭐
            (self.penalty_box_length, self.width / 2),  # 22. 大禁区长边中点 ⭐
        ]
        
        # 草皮分界点 (半场10.5块)
        # 全场等分步长由全场长度推导，便于复用到不同长度的吴川半场配置。
        # 半场范围内有 10 个完整分点
        full_length = self.length * 2
        step = full_length / 21
        
        extra_vertices = []
        for i in range(1, 11):  # 10个分点
            x = i * step
            if x < self.length:  # 确保在半场范围内
                extra_vertices.append((x, 0))          # 上边线分点
                extra_vertices.append((x, self.width)) # 下边线分点
        
        return base_vertices + required_points + extra_vertices

    edges: List[Tuple[int, int]] = field(default_factory=lambda: [
        (1, 2), (2, 3), (3, 4), (4, 5), (5, 6),
        (3, 7), (7, 8), (8, 4),
        (2, 10), (10, 11), (11, 12), (12, 13), (13, 5),
        (1, 14), (6, 17),
        (14, 15), (15, 19), (19, 16), (16, 17)
    ])

    @property
    def labels(self) -> List[str]:
        # 基础标签 (1-19)
        base_labels = [
            "01", "02", "03", "04", "05", "06", 
            "07", "08", "09", 
            "10", "11", "12", "13", 
            "14", "15", "16", "17", 
            "18", "19"
        ]
        
        # 必需关键点标签 (20-22)
        required_labels = ["20", "21", "22"]
        
        # 草皮分界点标签 (10上 + 10下 = 20个)
        extra_labels = []
        for i in range(1, 11):
            extra_labels.append(f"T{i}")  # Top
            extra_labels.append(f"B{i}")  # Bottom
        
        return base_labels + required_labels + extra_labels

    @property
    def colors(self) -> List[str]:
        # 基础颜色 (19个)
        base_colors = [
            "#FF1493", "#FF1493", "#FF1493", "#FF1493", "#FF1493", "#FF1493",
            "#FF1493", "#FF1493", "#FF1493", "#FF1493", "#FF1493", "#FF1493",
            "#FF1493", "#00BFFF", "#00BFFF", "#00BFFF", "#00BFFF", "#00BFFF",
            "#00BFFF"
        ]
        
        # 必需关键点颜色 (3个 - 白色突出显示)
        required_colors = ["#FFFFFF"] * 3
        
        # 草皮分界点颜色 (20个 - 绿色)
        extra_colors = ["#00FF00"] * 20
        
        return base_colors + required_colors + extra_colors


@dataclass
class HalfWuChuanYiZhongConfiguration(HalfWuChuanConfiguration):
    """吴川一中半场配置（34m × 46.5m）。"""

    width: int = 4650  # [cm] 场地宽 46.5m
    length: int = 3400  # [cm] 半场长 34m
    penalty_box_width: int = 2550  # [cm] 大禁区横向宽度 25.5m
    penalty_box_length: int = 995  # [cm] 大禁区向场内深度 9.95m
    goal_box_width: int = 1350  # [cm] 小禁区横向宽度 13.5m
    goal_box_length: int = 400  # [cm] 小禁区向场内深度 4m
    centre_circle_radius: int = 600  # [cm] 中圈半径 6m
    penalty_spot_distance: int = 900  # [cm] 点球点到底线 9m
