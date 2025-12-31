#!/usr/bin/env python3
"""
Markdown 格式检查脚本
检查 Markdown 文件的格式规范
"""

import os
import sys
import re
from pathlib import Path
from typing import List, Dict, Tuple


class FormatChecker:
    def __init__(self, root_dir: str):
        self.root_dir = Path(root_dir)
        self.errors = []
        self.warnings = []
        self.files_checked = 0

    def find_markdown_files(self) -> List[Path]:
        """查找所有 Markdown 文件"""
        md_files = []
        for path in self.root_dir.rglob("*.md"):
            # 跳过隐藏目录和 node_modules
            if any(part.startswith('.') for part in path.parts):
                continue
            if 'node_modules' in path.parts or 'vendor' in path.parts:
                continue
            md_files.append(path)
        return md_files

    def check_file_encoding(self, file_path: Path) -> bool:
        """检查文件编码是否为 UTF-8"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                f.read()
            return True
        except UnicodeDecodeError:
            self.errors.append(
                f"❌ {file_path.relative_to(self.root_dir)} - "
                f"文件编码不是 UTF-8"
            )
            return False

    def check_line_endings(self, file_path: Path, content: str) -> bool:
        """检查行尾符是否一致（LF）"""
        has_crlf = '\r\n' in content
        has_mixed = '\r\n' in content and '\n' in content.replace('\r\n', '')

        if has_mixed:
            self.errors.append(
                f"❌ {file_path.relative_to(self.root_dir)} - "
                f"行尾符混用（CRLF 和 LF）"
            )
            return False
        elif has_crlf:
            self.warnings.append(
                f"⚠️  {file_path.relative_to(self.root_dir)} - "
                f"使用 CRLF 行尾符，建议使用 LF"
            )

        return True

    def check_trailing_whitespace(self, file_path: Path, lines: List[str]) -> bool:
        """检查行尾空格"""
        has_trailing = False

        for i, line in enumerate(lines, 1):
            if line.rstrip('\n\r') != line.rstrip('\n\r').rstrip():
                self.warnings.append(
                    f"⚠️  {file_path.relative_to(self.root_dir)}:{i} - "
                    f"行尾有多余空格"
                )
                has_trailing = True

        return not has_trailing

    def check_file_ends_with_newline(self, file_path: Path, content: str) -> bool:
        """检查文件是否以换行符结尾"""
        if content and not content.endswith('\n'):
            self.warnings.append(
                f"⚠️  {file_path.relative_to(self.root_dir)} - "
                f"文件未以换行符结尾"
            )
            return False
        return True

    def check_heading_structure(self, file_path: Path, lines: List[str]) -> bool:
        """检查标题结构"""
        issues_found = False
        prev_level = 0
        has_h1 = False

        for i, line in enumerate(lines, 1):
            # 匹配标题
            heading_match = re.match(r'^(#{1,6})\s+(.+)$', line)

            if heading_match:
                level = len(heading_match.group(1))
                title = heading_match.group(2)

                # 检查是否有 H1
                if level == 1:
                    if has_h1:
                        self.warnings.append(
                            f"⚠️  {file_path.relative_to(self.root_dir)}:{i} - "
                            f"文件有多个 H1 标题"
                        )
                    has_h1 = True

                # 检查标题层级跳跃
                if prev_level > 0 and level > prev_level + 1:
                    self.warnings.append(
                        f"⚠️  {file_path.relative_to(self.root_dir)}:{i} - "
                        f"标题层级跳跃（从 H{prev_level} 到 H{level}）"
                    )
                    issues_found = True

                # 检查标题格式
                if not re.match(r'^#{1,6}\s+\S', line):
                    self.errors.append(
                        f"❌ {file_path.relative_to(self.root_dir)}:{i} - "
                        f"标题格式错误（# 后应有空格）"
                    )
                    issues_found = True

                prev_level = level

        # 检查是否有 H1 标题
        if not has_h1:
            self.warnings.append(
                f"⚠️  {file_path.relative_to(self.root_dir)} - "
                f"文件缺少 H1 标题"
            )

        return not issues_found

    def check_code_blocks(self, file_path: Path, lines: List[str]) -> bool:
        """检查代码块格式"""
        issues_found = False
        in_code_block = False
        code_block_start = 0

        for i, line in enumerate(lines, 1):
            # 检查代码块标记
            if line.strip().startswith('```'):
                if in_code_block:
                    in_code_block = False
                else:
                    in_code_block = True
                    code_block_start = i

                    # 检查是否指定了语言
                    if line.strip() == '```':
                        self.warnings.append(
                            f"⚠️  {file_path.relative_to(self.root_dir)}:{i} - "
                            f"代码块未指定语言"
                        )

        # 检查是否有未闭合的代码块
        if in_code_block:
            self.errors.append(
                f"❌ {file_path.relative_to(self.root_dir)}:{code_block_start} - "
                f"代码块未闭合"
            )
            issues_found = True

        return not issues_found

    def check_link_format(self, file_path: Path, lines: List[str]) -> bool:
        """检查链接格式"""
        issues_found = False

        for i, line in enumerate(lines, 1):
            # 检查损坏的链接格式
            if re.search(r'\]\s*\(', line):
                if re.search(r'\]\s+\(', line):
                    self.errors.append(
                        f"❌ {file_path.relative_to(self.root_dir)}:{i} - "
                        f"链接格式错误（] 和 ( 之间不应有空格）"
                    )
                    issues_found = True

            # 检查图片链接
            if re.search(r'!\[.*\]\(.*\)', line):
                # 检查图片是否有 alt 文本
                if re.search(r'!\[\]\(', line):
                    self.warnings.append(
                        f"⚠️  {file_path.relative_to(self.root_dir)}:{i} - "
                        f"图片缺少 alt 文本"
                    )

        return not issues_found

    def check_list_format(self, file_path: Path, lines: List[str]) -> bool:
        """检查列表格式"""
        issues_found = False

        for i, line in enumerate(lines, 1):
            # 检查无序列表
            if re.match(r'^[\*\+\-]\s', line):
                # 检查缩进
                if not re.match(r'^(  )*[\*\+\-]\s', line):
                    self.warnings.append(
                        f"⚠️  {file_path.relative_to(self.root_dir)}:{i} - "
                        f"列表缩进不规范（建议使用2个空格）"
                    )

            # 检查有序列表
            if re.match(r'^\d+\.\s', line):
                # 检查是否使用了正确的数字序号
                pass  # 可以添加更多检查

        return not issues_found

    def check_chinese_punctuation(self, file_path: Path, lines: List[str]) -> bool:
        """检查中英文混排的标点符号"""
        issues_found = False

        for i, line in enumerate(lines, 1):
            # 跳过代码块
            if line.strip().startswith('```') or line.strip().startswith('    '):
                continue

            # 检查中文后面使用英文标点
            if re.search(r'[\u4e00-\u9fff][,\.;:!?]', line):
                self.warnings.append(
                    f"⚠️  {file_path.relative_to(self.root_dir)}:{i} - "
                    f"中文后使用了英文标点符号"
                )

            # 检查英文和中文之间是否有空格
            # if re.search(r'[a-zA-Z][\u4e00-\u9fff]|[\u4e00-\u9fff][a-zA-Z]', line):
            #     self.warnings.append(
            #         f"⚠️  {file_path.relative_to(self.root_dir)}:{i} - "
            #         f"建议在中英文之间添加空格"
            #     )

        return not issues_found

    def check_file(self, file_path: Path) -> bool:
        """检查单个文件"""
        # 检查文件编码
        if not self.check_file_encoding(file_path):
            return False

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                lines = content.splitlines(keepends=True)

        except Exception as e:
            self.errors.append(
                f"❌ {file_path.relative_to(self.root_dir)} - "
                f"读取文件失败: {e}"
            )
            return False

        # 执行各项检查
        checks = [
            self.check_line_endings(file_path, content),
            self.check_trailing_whitespace(file_path, lines),
            self.check_file_ends_with_newline(file_path, content),
            self.check_heading_structure(file_path, lines),
            self.check_code_blocks(file_path, lines),
            self.check_link_format(file_path, lines),
            self.check_list_format(file_path, lines),
            self.check_chinese_punctuation(file_path, lines),
        ]

        return all(checks)

    def run(self) -> int:
        """运行格式检查"""
        print("🔍 开始 Markdown 格式检查...")
        print(f"📂 扫描目录: {self.root_dir}")
        print()

        md_files = self.find_markdown_files()
        print(f"📄 找到 {len(md_files)} 个 Markdown 文件")
        print()

        for file_path in md_files:
            self.check_file(file_path)
            self.files_checked += 1

        print("=" * 70)
        print("📊 检查结果:")
        print("=" * 70)
        print(f"✅ 检查文件数: {self.files_checked}")
        print(f"❌ 错误: {len(self.errors)}")
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

        if self.errors:
            print("💥 格式检查失败！请修复上述错误。")
            return 1
        elif self.warnings:
            print("⚠️  格式检查通过，但有警告。建议修复以提高文档质量。")
            return 0
        else:
            print("✨ 所有格式检查通过！")
            return 0


def main():
    import argparse

    parser = argparse.ArgumentParser(description='检查 Markdown 文件格式')
    parser.add_argument(
        'directory',
        nargs='?',
        default='.',
        help='要检查的目录（默认：当前目录）'
    )

    args = parser.parse_args()

    root_dir = os.path.abspath(args.directory)

    if not os.path.isdir(root_dir):
        print(f"错误：{root_dir} 不是有效目录")
        return 1

    checker = FormatChecker(root_dir)
    return checker.run()


if __name__ == '__main__':
    sys.exit(main())
