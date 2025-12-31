#!/usr/bin/env python3
"""
链接验证脚本
验证 Markdown 文件中的内部和外部链接是否有效
"""

import argparse
import os
import re
import sys
from pathlib import Path
from typing import List, Tuple, Set
from urllib.parse import urlparse, urljoin
import concurrent.futures

try:
    import requests
    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry
except ImportError:
    print("错误：需要安装 requests 库")
    print("运行：pip install requests")
    sys.exit(1)


class LinkValidator:
    def __init__(self, root_dir: str, internal_only: bool = False, external_only: bool = False):
        self.root_dir = Path(root_dir)
        self.internal_only = internal_only
        self.external_only = external_only
        self.errors = []
        self.warnings = []
        self.checked_urls = {}  # 缓存已检查的 URL

        # 配置 requests session
        self.session = requests.Session()
        retry_strategy = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504]
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (compatible; LinkValidator/1.0)'
        })

    def find_markdown_files(self) -> List[Path]:
        """查找所有 Markdown 文件"""
        md_files = []
        for path in self.root_dir.rglob("*.md"):
            # 跳过特定目录
            if any(part.startswith('.') for part in path.parts):
                continue
            if 'node_modules' in path.parts or 'vendor' in path.parts:
                continue
            md_files.append(path)
        return md_files

    def extract_links(self, file_path: Path) -> List[Tuple[str, int]]:
        """从 Markdown 文件中提取所有链接"""
        links = []

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.readlines()

            for line_num, line in enumerate(content, 1):
                # 匹配 Markdown 链接：[text](url)
                markdown_links = re.finditer(r'\[([^\]]+)\]\(([^)]+)\)', line)
                for match in markdown_links:
                    url = match.group(2)
                    links.append((url, line_num))

                # 匹配 HTML 链接：<a href="url">
                html_links = re.finditer(r'<a\s+(?:[^>]*?\s+)?href="([^"]*)"', line)
                for match in html_links:
                    url = match.group(1)
                    links.append((url, line_num))

                # 匹配直接 URL
                url_pattern = re.finditer(r'https?://[^\s<>"{}|\\^`\[\]]+', line)
                for match in url_pattern:
                    url = match.group(0)
                    links.append((url, line_num))

        except Exception as e:
            self.errors.append(f"读取文件失败 {file_path}: {e}")

        return links

    def is_external_link(self, url: str) -> bool:
        """判断是否为外部链接"""
        return url.startswith(('http://', 'https://'))

    def validate_internal_link(self, file_path: Path, link: str, line_num: int) -> bool:
        """验证内部链接"""
        # 跳过锚点链接
        if link.startswith('#'):
            return True

        # 移除锚点
        link_without_anchor = link.split('#')[0]
        if not link_without_anchor:
            return True

        # 计算目标文件路径
        if link_without_anchor.startswith('/'):
            # 绝对路径
            target_path = self.root_dir / link_without_anchor.lstrip('/')
        else:
            # 相对路径
            target_path = (file_path.parent / link_without_anchor).resolve()

        if not target_path.exists():
            self.errors.append(
                f"❌ {file_path.relative_to(self.root_dir)}:{line_num} - "
                f"内部链接失效: {link}"
            )
            return False

        return True

    def validate_external_link(self, file_path: Path, url: str, line_num: int) -> bool:
        """验证外部链接"""
        # 检查缓存
        if url in self.checked_urls:
            return self.checked_urls[url]

        try:
            response = self.session.head(url, timeout=10, allow_redirects=True)

            # 如果 HEAD 失败，尝试 GET
            if response.status_code >= 400:
                response = self.session.get(url, timeout=10, allow_redirects=True)

            is_valid = response.status_code < 400

            if not is_valid:
                self.errors.append(
                    f"❌ {file_path.relative_to(self.root_dir)}:{line_num} - "
                    f"外部链接失效 (HTTP {response.status_code}): {url}"
                )

            self.checked_urls[url] = is_valid
            return is_valid

        except requests.exceptions.Timeout:
            self.warnings.append(
                f"⚠️  {file_path.relative_to(self.root_dir)}:{line_num} - "
                f"链接超时: {url}"
            )
            self.checked_urls[url] = False
            return False

        except requests.exceptions.RequestException as e:
            self.errors.append(
                f"❌ {file_path.relative_to(self.root_dir)}:{line_num} - "
                f"无法访问链接: {url} ({str(e)})"
            )
            self.checked_urls[url] = False
            return False

    def validate_file(self, file_path: Path) -> Tuple[int, int]:
        """验证单个文件中的所有链接"""
        links = self.extract_links(file_path)
        valid_count = 0
        invalid_count = 0

        for link, line_num in links:
            if self.is_external_link(link):
                if not self.internal_only:
                    if self.validate_external_link(file_path, link, line_num):
                        valid_count += 1
                    else:
                        invalid_count += 1
            else:
                if not self.external_only:
                    if self.validate_internal_link(file_path, link, line_num):
                        valid_count += 1
                    else:
                        invalid_count += 1

        return valid_count, invalid_count

    def run(self) -> int:
        """运行链接验证"""
        print("🔍 开始链接验证...")
        print(f"📂 扫描目录: {self.root_dir}")

        if self.internal_only:
            print("🔗 仅检查内部链接")
        elif self.external_only:
            print("🌐 仅检查外部链接")
        else:
            print("🔗 检查所有链接")

        print()

        md_files = self.find_markdown_files()
        print(f"📄 找到 {len(md_files)} 个 Markdown 文件")
        print()

        total_valid = 0
        total_invalid = 0

        # 使用进度条
        for i, file_path in enumerate(md_files, 1):
            print(f"[{i}/{len(md_files)}] 检查: {file_path.relative_to(self.root_dir)}")
            valid, invalid = self.validate_file(file_path)
            total_valid += valid
            total_invalid += invalid

        print("\n" + "=" * 70)
        print("📊 验证结果:")
        print("=" * 70)
        print(f"✅ 有效链接: {total_valid}")
        print(f"❌ 失效链接: {total_invalid}")
        print(f"⚠️  警告: {len(self.warnings)}")
        print()

        if self.errors:
            print("❌ 发现的错误:")
            for error in self.errors:
                print(f"  {error}")
            print()

        if self.warnings:
            print("⚠️  警告:")
            for warning in self.warnings:
                print(f"  {warning}")
            print()

        if total_invalid > 0 or self.errors:
            print("💥 链接验证失败！")
            return 1
        else:
            print("✨ 所有链接验证通过！")
            return 0


def main():
    parser = argparse.ArgumentParser(description='验证 Markdown 文件中的链接')
    parser.add_argument(
        'directory',
        nargs='?',
        default='.',
        help='要检查的目录（默认：当前目录）'
    )
    parser.add_argument(
        '--internal-only',
        action='store_true',
        help='仅检查内部链接'
    )
    parser.add_argument(
        '--external',
        action='store_true',
        help='仅检查外部链接'
    )

    args = parser.parse_args()

    root_dir = os.path.abspath(args.directory)

    if not os.path.isdir(root_dir):
        print(f"错误：{root_dir} 不是有效目录")
        return 1

    validator = LinkValidator(
        root_dir,
        internal_only=args.internal_only,
        external_only=args.external
    )

    return validator.run()


if __name__ == '__main__':
    sys.exit(main())
