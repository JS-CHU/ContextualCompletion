import vtk
import numpy as np
import os
import pickle

def render(points, scalars, r, path):
    def save_as_vector_image():
        exporter = vtk.vtkGL2PSExporter()
        exporter.SetInput(render_window)
        exporter.SetFileFormatToSVG()  # 或者使用 SetFileFormatToPDF(), SetFileFormatToEPS()
        exporter.SetFilePrefix(os.path.join(path, 'rendered_P'))  # 保存的文件名前缀
        exporter.CompressOff()  # 关闭压缩，可以选择性开启
        exporter.Write()
        print("Image saved as manual_rotated_image.svg")

    # 添加一个键盘事件，当按下 's' 键时保存图像
    def on_key_press(obj, event):
        key = obj.GetKeySym()
        if key == 's':
            save_as_vector_image()
    # 创建一个点云
    vtk_points = vtk.vtkPoints()
    for i, point in enumerate(points):
        vtk_points.InsertNextPoint(point)

    # 创建一个球体的Glyph，并提高其分辨率
    sphere = vtk.vtkSphereSource()
    sphere.SetRadius(r)  # 设置球的半径
    sphere.SetThetaResolution(32)  # 增加theta方向的分辨率
    sphere.SetPhiResolution(32)  # 增加phi方向的分辨率

    # 创建PolyData来存储点和标量
    poly_data = vtk.vtkPolyData()
    poly_data.SetPoints(vtk_points)

    # 创建标量数组并设置
    scalar_values = vtk.vtkFloatArray()
    for scalar in scalars:
        scalar_values.InsertNextValue(scalar)
    poly_data.GetPointData().SetScalars(scalar_values)

    # 使用Glyph3D来表示球体
    glyph = vtk.vtkGlyph3D()
    glyph.SetSourceConnection(sphere.GetOutputPort())
    glyph.SetInputData(poly_data)
    glyph.SetColorModeToColorByScalar()
    glyph.ScalingOff()  # 关闭缩放，所有球体同样大小
    glyph.Update()

    # 创建自定义颜色映射（从黄色到红色）
    lut = vtk.vtkLookupTable()
    lut.SetNumberOfTableValues(256)
    lut.Build()

    # blue -> red
    # for i in range(256):
    #     r = i / 255.0  # 红色分量，标量值越大越红
    #     g = 0.2  # 绿色分量，标量值越大越少绿
    #     b = (255 - i) / 255.0  # 蓝色分量为0
    #     lut.SetTableValue(i, r, g, b, 1.0)  # RGBA值

    # yellow -> red
    # for i in range(256):
    #     r = 1.0  # 红色分量固定为1.0
    #     g = (255 - i) / 255.0  # 绿色分量从1.0逐渐减少到0.0
    #     b = 0.1  # 蓝色分量固定为0.0
    #     lut.SetTableValue(i, r, g, b, 1.0)  # RGBA值

    # blue -> yellow
    # for i in range(256):
    #     r = i / 255.0  # 红色分量从 0.0 逐渐增加到 1.0
    #     g = i / 255.0  # 绿色分量从 0.0 逐渐增加到 1.0
    #     b = ((255 - i) / 255.0) * 0.5  # 蓝色分量从 1.0 逐渐减少到 0.0
    #     lut.SetTableValue(i, r, g, b, 1.0)  # RGBA值

    # # brown -> yellow
    # for i in range(256):
    #     r = 0.5 + 0.4 * (i / 255.0)  # 红色分量固定为1.0
    #     g = 0.4 + 0.6 * (i / 255.0)  # 绿色分量从1.0逐渐减少到0.0
    #     b = 0.5  # 蓝色分量固定为0.0
    #     lut.SetTableValue(i, r, g, b, 1.0)  # RGBA值

    # purple -> yellow
    for i in range(256):
        r = 0.75  # 红色分量固定为1.0
        g = 0.75  # 绿色分量从1.0逐渐减少到0.0
        b = (255 - i) / 255.0  # 蓝色分量固定为0.0
        lut.SetTableValue(i, r, g, b, 1.0)  # RGBA值


    # 创建一个Mapper
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputConnection(glyph.GetOutputPort())
    mapper.SetLookupTable(lut)  # 使用自定义的颜色映射表
    mapper.SetScalarRange(scalars.min(), scalars.max())

    # 创建一个Actor
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)

    # 渲染场景
    renderer = vtk.vtkRenderer()
    renderer.AddActor(actor)
    renderer.SetBackground(1, 1, 1)  # 背景色为白色

    # 创建RenderWindow
    render_window = vtk.vtkRenderWindow()
    render_window.AddRenderer(renderer)

    # 创建RenderWindowInteractor
    render_window_interactor = vtk.vtkRenderWindowInteractor()
    render_window_interactor.SetRenderWindow(render_window)

    # 开始渲染
    render_window.Render()
    # 绑定按键事件
    render_window_interactor.AddObserver('KeyPressEvent', on_key_press)
    render_window_interactor.Start()


def render_gt(points, index, r, path):
    def save_as_vector_image(path):
        exporter = vtk.vtkGL2PSExporter()
        exporter.SetInput(render_window)
        exporter.SetFileFormatToSVG()  # 或者使用 SetFileFormatToPDF(), SetFileFormatToEPS()
        exporter.SetFilePrefix(os.path.join(path, 'rendered_gt'))  # 保存的文件名前缀
        exporter.CompressOff()  # 关闭压缩，可以选择性开启
        exporter.Write()
        print("Image saved as manual_rotated_image.svg")

    # 添加一个键盘事件，当按下 's' 键时保存图像
    def on_key_press(obj, event):
        key = obj.GetKeySym()
        if key == 's':
            save_as_vector_image(path)
    colors = vtk.vtkUnsignedCharArray()
    colors.SetNumberOfComponents(3)  # RGB颜色
    colors.SetName("Colors")
    # 创建一个点云
    vtk_points = vtk.vtkPoints()
    for i, point in enumerate(points):
        vtk_points.InsertNextPoint(point)
        # 默认颜色为黄色
        color = [180, 180, 255]  # 黄色 (RGB: 255, 255, 0)
        colors.InsertNextTuple(color)

    # 按索引将某个点渲染为红色
    colors.SetTuple(index, [180, 255, 0])  # 设置为红色 (RGB: 255, 0, 0)

    # 创建PolyData来存储点和颜色
    poly_data = vtk.vtkPolyData()
    poly_data.SetPoints(vtk_points)
    poly_data.GetPointData().SetScalars(colors)

    # 创建球体的Glyph
    sphere = vtk.vtkSphereSource()
    sphere.SetRadius(r)  # 设置球的半径
    sphere.SetThetaResolution(32)
    sphere.SetPhiResolution(32)

    # 使用Glyph3D将点渲染成球体
    glyph = vtk.vtkGlyph3D()
    glyph.SetSourceConnection(sphere.GetOutputPort())
    glyph.SetInputData(poly_data)
    glyph.SetColorModeToColorByScalar()  # 使用设置的颜色
    glyph.ScalingOff()  # 关闭缩放
    glyph.Update()

    # 创建Mapper
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputConnection(glyph.GetOutputPort())

    # 创建Actor
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)

    # 创建Renderer
    renderer = vtk.vtkRenderer()
    renderer.AddActor(actor)
    renderer.SetBackground(1, 1, 1)  # 背景设置为白色

    # 创建RenderWindow
    render_window = vtk.vtkRenderWindow()
    render_window.AddRenderer(renderer)

    # 创建RenderWindowInteractor并启用交互
    render_window_interactor = vtk.vtkRenderWindowInteractor()
    render_window_interactor.SetRenderWindow(render_window)

    # 开始渲染并启动交互
    render_window.Render()
    # 绑定按键事件
    render_window_interactor.AddObserver('KeyPressEvent', on_key_press)
    render_window_interactor.Start()


def set_value(data):
    # 步骤1: 找到最小的1984个元素的最大值
    # 获取排序后的索引
    sorted_indices = np.argsort(data)
    # 最小的1984个元素的索引
    min_1984_indices = sorted_indices[:1984]
    # 找到这些元素中的最大值
    max_of_min_1984 = data[min_1984_indices].max()

    # 步骤2: 找到最大的64个元素的索引
    max_64_indices = sorted_indices[-64:]

    # 步骤3: 创建在一倍最大值到两倍最大值之间的等差数列
    # 创建等差数列
    arithmetic_sequence = np.linspace(max_of_min_1984, 2 * max_of_min_1984, 64)

    # 按原来的大小顺序设置这些最大值对应的元素
    # 获取64个最大值的原始顺序
    sorted_max_64_indices = np.argsort(data[max_64_indices])
    # 重新排序这些最大值
    data[max_64_indices[sorted_max_64_indices]] = arithmetic_sequence
    return data

def render_pc(points, r, path):
    def save_as_vector_image(path):
        exporter = vtk.vtkGL2PSExporter()
        exporter.SetInput(render_window)
        exporter.SetFileFormatToSVG()  # 或者使用 SetFileFormatToPDF(), SetFileFormatToEPS()
        exporter.SetFilePrefix(os.path.join(path, 'rendered_gt'))  # 保存的文件名前缀
        exporter.CompressOff()  # 关闭压缩，可以选择性开启
        exporter.Write()
        print("Image saved as manual_rotated_image.svg")

    # 添加一个键盘事件，当按下 's' 键时保存图像
    def on_key_press(obj, event):
        key = obj.GetKeySym()
        if key == 's':
            save_as_vector_image(path)
    colors = vtk.vtkUnsignedCharArray()
    colors.SetNumberOfComponents(3)  # RGB颜色
    colors.SetName("Colors")
    # 创建一个点云
    vtk_points = vtk.vtkPoints()
    for i, point in enumerate(points):
        vtk_points.InsertNextPoint(point)
        # 默认颜色为黄色
        color = [192, 192, 0]  # 黄色 (RGB: 255, 255, 0)
        colors.InsertNextTuple(color)

    # 创建PolyData来存储点和颜色
    poly_data = vtk.vtkPolyData()
    poly_data.SetPoints(vtk_points)
    poly_data.GetPointData().SetScalars(colors)

    # 创建球体的Glyph
    sphere = vtk.vtkSphereSource()
    sphere.SetRadius(r)  # 设置球的半径
    sphere.SetThetaResolution(32)
    sphere.SetPhiResolution(32)

    # 使用Glyph3D将点渲染成球体
    glyph = vtk.vtkGlyph3D()
    glyph.SetSourceConnection(sphere.GetOutputPort())
    glyph.SetInputData(poly_data)
    glyph.SetColorModeToColorByScalar()  # 使用设置的颜色
    glyph.ScalingOff()  # 关闭缩放
    glyph.Update()

    # 创建Mapper
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputConnection(glyph.GetOutputPort())

    # 创建Actor
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)

    # 创建Renderer
    renderer = vtk.vtkRenderer()
    renderer.AddActor(actor)
    renderer.SetBackground(1, 1, 1)  # 背景设置为白色

    # 创建RenderWindow
    render_window = vtk.vtkRenderWindow()
    render_window.AddRenderer(renderer)

    # 创建RenderWindowInteractor并启用交互
    render_window_interactor = vtk.vtkRenderWindowInteractor()
    render_window_interactor.SetRenderWindow(render_window)

    # 开始渲染并启动交互
    render_window.Render()
    # 绑定按键事件
    render_window_interactor.AddObserver('KeyPressEvent', on_key_press)
    render_window_interactor.Start()

def render_find_similar():
    dir = ''
    # 示例点云数据
    points = np.loadtxt(os.path.join(dir, 'P.xyz'))
    # gt = np.loadtxt(os.path.join(dir, 'gt.xyz'))
    # cat = np.loadtxt(os.path.join(dir, 'q_cat.xyz'))
    refine = np.loadtxt(os.path.join(dir, 'q_refine.xyz'))
    scalars_np = np.loadtxt(os.path.join(dir, 'w.txt'))
    step =0
    for j in range(266, 2048):
        scalars = scalars_np[j]
        print(j)
        # scalars = np.log1p(np.log1p(np.log1p(scalars_np[j])))

        # transformer = QuantileTransformer(output_distribution='uniform')
        # transformed_data = transformer.fit_transform(scalars.reshape(-1, 1)).flatten()

        scalars = set_value(scalars_np[j])
        render_gt(refine, j, r=0.07, path=dir)
        render(points, scalars, r=0.04, path=dir)
        step += 1
        if step == 19:
            exit()

def render_similar():
    dir = ''
    # 示例点云数据
    points = np.loadtxt(os.path.join(dir, 'P.xyz'))
    # gt = np.loadtxt(os.path.join(dir, 'gt.xyz'))
    # cat = np.loadtxt(os.path.join(dir, 'q_cat.xyz'))
    refine = np.loadtxt(os.path.join(dir, 'q_refine.xyz'))
    scalars_np = np.loadtxt(os.path.join(dir, 'w.txt'))
    step = 0
    for j in range(266, 2048):
        scalars = scalars_np[j]
        print(j)
        # scalars = np.log1p(np.log1p(np.log1p(scalars_np[j])))

        # transformer = QuantileTransformer(output_distribution='uniform')
        # transformed_data = transformer.fit_transform(scalars.reshape(-1, 1)).flatten()

        scalars = set_value(scalars_np[j])
        render_gt(refine, j, r=0.07, path=dir)
        render(points, scalars, r=0.04, path=dir)
        step += 1
        if step == 19:
            exit()

def render_p_gen():
    dir = ''
    # 示例点云数据
    path = ''
    with open(path, 'rb') as f:
        points = pickle.load(f).astype(np.float32)
    # points = np.loadtxt(os.path.join(dir, 'P.xyz'))
    gen = np.loadtxt(os.path.join(dir, 'gen.xyz'))
    render_pc(gen, r=0.01, path=dir)
    render_pc(points, r=0.01, path=dir)

render_p_gen()