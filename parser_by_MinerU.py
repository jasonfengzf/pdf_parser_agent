import requests
import os
import time
import zipfile
from config import MINERU_TOKEN

# 配置参数
FILE_FOLDER = "pdf_file"  # 文件文件夹路径
OUTPUT_FOLDER = "output_zip"  # 下载结果保存路径
EXTRACT_FOLDER = "output_extracted"  # 解压文件保存路径
CHECK_INTERVAL = 10  # 检查处理状态的间隔时间（秒）
MAX_WAIT_TIME = 300  # 最大等待时间（秒）

# 确保输出文件夹存在
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
os.makedirs(EXTRACT_FOLDER, exist_ok=True)


def get_files_from_folder(folder_path):
    """从文件夹获取所有支持的文件"""
    supported_extensions = ['.pdf', '.doc', '.docx', '.ppt', '.pptx', '.txt']
    files = []

    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if os.path.isfile(file_path) and any(filename.lower().endswith(ext) for ext in supported_extensions):
            files.append({
                "name": filename,
                "path": file_path,
                "is_ocr": True,
                "data_id": f"file_{len(files)}"  # 生成唯一的data_id
            })

    return files


def apply_upload_urls(files_info):
    """申请文件上传URL"""
    url = "https://mineru.net/api/v4/file-urls/batch"
    header = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {MINERU_TOKEN}"
    }

    data = {
        "enable_formula": True,
        "enable_table": True,
        "files": files_info
    }

    try:
        response = requests.post(url, headers=header, json=data)
        if response.status_code == 200:
            result = response.json()
            print('申请上传URL成功')
            if result["code"] == 0:
                batch_id = result["data"]["batch_id"]
                urls = result["data"]["file_urls"]
                print(f'batch_id: {batch_id}, 文件数量: {len(urls)}')
                return batch_id, urls
            else:
                print(f'申请上传URL失败, 原因: {result.get("msg", "未知错误")}')
                return None, None
        else:
            print(f'响应不成功, 状态码: {response.status_code}, 结果: {response.text}')
            return None, None
    except Exception as err:
        print(f'申请上传URL时发生错误: {err}')
        return None, None


def upload_files(files_info, urls):
    """上传文件到指定URL"""
    print("开始上传文件...")
    success_count = 0

    for i, file_info in enumerate(files_info):
        if i < len(urls):
            try:
                with open(file_info["path"], 'rb') as f:
                    res_upload = requests.put(urls[i], data=f)
                    if res_upload.status_code == 200:
                        print(f"{file_info['name']} 上传成功")
                        success_count += 1
                    else:
                        print(f"{file_info['name']} 上传失败, 状态码: {res_upload.status_code}")
            except Exception as e:
                print(f"上传文件 {file_info['name']} 时发生错误: {e}")
        else:
            print(f"文件 {file_info['name']} 没有对应的上传URL")

    print(f"文件上传完成，成功上传 {success_count}/{len(files_info)} 个文件")
    return success_count


def wait_for_processing(batch_id, max_wait_time=MAX_WAIT_TIME):
    """等待处理完成"""
    url = f"https://mineru.net/api/v4/extract-results/batch/{batch_id}"
    header = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {MINERU_TOKEN}"
    }

    print("等待处理完成...")
    start_time = time.time()

    while time.time() - start_time < max_wait_time:
        try:
            res = requests.get(url, headers=header)
            if res.status_code == 200:
                result = res.json()
                if result["code"] == 0:
                    extract_results = result["data"]["extract_result"]

                    # 检查所有文件是否处理完成
                    all_done = True
                    for file_result in extract_results:
                        state = file_result["state"]
                        if state == "done":
                            print(f"文件 {file_result['file_name']} 处理完成")
                        elif state == "processing":
                            print(f"文件 {file_result['file_name']} 正在处理中...")
                            all_done = False
                        elif state == "failed":
                            print(f"文件 {file_result['file_name']} 处理失败: {file_result.get('err_msg', '未知错误')}")
                        else:
                            print(f"文件 {file_result['file_name']} 状态: {state}")
                            all_done = False

                    if all_done:
                        print("所有文件处理完成！")
                        return extract_results

            else:
                print(f"查询处理状态失败，状态码: {res.status_code}")

        except Exception as e:
            print(f"查询处理状态时发生错误: {e}")

        # 等待一段时间再检查
        print(f"等待 {CHECK_INTERVAL} 秒后再次检查...")
        time.sleep(CHECK_INTERVAL)

    print("超过最大等待时间，处理可能未完成")
    return None


def download_results(extract_results):
    """下载处理结果"""
    print("开始下载处理结果...")
    success_count = 0
    downloaded_files = []

    for result in extract_results:
        if result["state"] == "done" and "full_zip_url" in result:
            try:
                download_url = result["full_zip_url"]
                filename = result["file_name"]
                output_filename = f"{os.path.splitext(filename)[0]}_result.zip"
                output_path = os.path.join(OUTPUT_FOLDER, output_filename)

                # 下载文件
                response = requests.get(download_url, stream=True)
                if response.status_code == 200:
                    with open(output_path, 'wb') as f:
                        for chunk in response.iter_content(chunk_size=8192):
                            f.write(chunk)
                    print(f"下载成功: {output_filename}")
                    success_count += 1
                    downloaded_files.append(output_path)
                else:
                    print(f"下载失败 {filename}: 状态码 {response.status_code}")

            except Exception as e:
                print(f"下载文件 {result['file_name']} 时发生错误: {e}")
        else:
            print(f"文件 {result['file_name']} 未完成处理或没有下载链接")

    print(f"下载完成，成功下载 {success_count}/{len(extract_results)} 个文件")
    return downloaded_files


def extract_zip_files(zip_files):
    """解压ZIP文件到对应的独立文件夹"""
    print("开始解压文件...")
    success_count = 0

    for zip_path in zip_files:
        try:
            # 创建对应的解压文件夹
            zip_filename = os.path.basename(zip_path)
            extract_folder_name = os.path.splitext(zip_filename)[0]  # 去掉.zip后缀
            extract_path = os.path.join(EXTRACT_FOLDER, extract_folder_name)

            # 如果文件夹已存在，先删除
            if os.path.exists(extract_path):
                import shutil
                shutil.rmtree(extract_path)

            os.makedirs(extract_path, exist_ok=True)

            # 解压文件
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(extract_path)

            # 统计解压出的文件
            extracted_files = []
            for root, dirs, files in os.walk(extract_path):
                for file in files:
                    extracted_files.append(file)

            print(f"解压成功: {zip_filename} -> {extract_folder_name}/ (包含 {len(extracted_files)} 个文件)")
            success_count += 1

        except zipfile.BadZipFile:
            print(f"解压失败: {zip_path} - 文件损坏或不是有效的ZIP文件")
        except Exception as e:
            print(f"解压文件 {zip_path} 时发生错误: {e}")

    print(f"解压完成，成功解压 {success_count}/{len(zip_files)} 个ZIP文件")
    return success_count


def parser_by_MinerU():
    """主函数"""
    print("开始批量处理文件...")

    # 1. 获取文件夹中的文件
    files_info = get_files_from_folder(FILE_FOLDER)
    if not files_info:
        print(f"在文件夹 {FILE_FOLDER} 中没有找到支持的文件")
        return

    print(f"找到 {len(files_info)} 个文件: {[f['name'] for f in files_info]}")

    # 2. 申请上传URL
    batch_id, urls = apply_upload_urls(files_info)
    if not batch_id or not urls:
        print("申请上传URL失败，程序退出")
        return

    # 3. 上传文件
    success_count = upload_files(files_info, urls)
    if success_count == 0:
        print("没有文件上传成功，程序退出")
        return

    # 4. 等待处理完成
    extract_results = wait_for_processing(batch_id)
    if not extract_results:
        print("处理未完成或查询失败，程序退出")
        return

    # 5. 下载结果
    downloaded_files = download_results(extract_results)
    if not downloaded_files:
        print("没有文件下载成功，程序退出")
        return

    # 6. 解压文件
    extract_success = extract_zip_files(downloaded_files)

    print("批量处理完成！")
    print(f"处理总结:")
    print(f"  - 上传文件: {success_count}/{len(files_info)}")
    print(f"  - 下载结果: {len(downloaded_files)}/{len(extract_results)}")
    print(f"  - 解压文件: {extract_success}/{len(downloaded_files)}")
    print(f"文件保存位置:")
    print(f"  - 原始ZIP文件: {OUTPUT_FOLDER}/")
    print(f"  - 解压文件: {EXTRACT_FOLDER}/")


if __name__ == "__main__":
    parser_by_MinerU()