from PIL import Image

def split_and_save_digits(image_path, output_folder):
    # 打开图片
    image = Image.open(image_path)

    # 获取图片尺寸
    width, height = image.size

    # 每个数字的宽度
    digit_width = width // 10

    # 创建输出文件夹
    os.makedirs(output_folder, exist_ok=True)

    # 分割数字并保存
    for i in range(10):
        left = i * digit_width
        right = (i + 1) * digit_width

        # 分割图像
        digit_image = image.crop((left, 0, right, height))

        # 保存分割后的数字图片
        digit_image.save(os.path.join(output_folder, f"{i}.png"))

if __name__ == "__main__":
    import os

    # 输入图片路径和输出文件夹路径
    image_path = r"assinment_3\`WB~PDH%AMPJI4L2I6VVM54.png"
    output_folder = r"assinment_3/split"

    # 执行分割与保存
    split_and_save_digits(image_path, output_folder)

    print("分割完成，数字图片已保存到指定文件夹。")
