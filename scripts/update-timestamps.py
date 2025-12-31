#!/usr/bin/env python3
"""
时间戳更新脚本
自动更新 Markdown 文件的时间戳信息
"""

import os
import sys
import re
from pathlib import Path
from datetime import datetime
from typing import List, Optional, Tuple
import subprocess


class TimestampUpdater:
    def __init__(self, root_dir: str, use_git: bool = True):
        self.root_dir = Path(root_dir)
        self.use_git = use_git
        self.updated_files = []
        self.skipped_files = []

    def find_markdown_files(self) -> List[Path]:
        """查找所有 Markdown 文件"""
        md_files = []
        for path in self.root_dir.rglob("*.md"):
            # 跳过隐藏目录和特殊目录
            if any(part.startswith('.') for part in path.parts):
                continue
            if 'node_modules' in path.parts or 'vendor' in path.parts:
                continue
            md_files.append(path)
        return md_files

    def get_git_timestamp(self, file_path: Path) -> Optional[Tuple[datetime, datetime]]:
        """从 Git 获取文件的创建和修改时间"""
        if not self.use_git:
            return None

        try:
            # 获取第一次提交时间（创建时间）
            created_cmd = [
                'git', 'log', '--follow', '--format=%aI', '--reverse',
                str(file_path.relative_to(self.root_dir))
            ]
            created_result = subprocess.run(
                created_cmd,
                cwd=self.root_dir,
                capture_output=True,
                text=True,
                check=True
            )

            # 获取最后一次提交时间（修改时间）
            modified_cmd = [
                'git', 'log', '-1', '--format=%aI',
                str(file_path.relative_to(self.root_dir))
            ]
            modified_result = subprocess.run(
                modified_cmd,
                cwd=self.root_dir,
                capture_output=True,
                text=True,
                check=True
            )

            created_str = created_result.stdout.strip().split('\n')[0] if created_result.stdout.strip() else None
            modified_str = modified_result.stdout.strip()

            if created_str and modified_str:
                created = datetime.fromisoformat(created_str.replace('Z', '+00:00'))
                modified = datetime.fromisoformat(modified_str.replace('Z', '+00:00'))
                return (created, modified)

        except (subprocess.CalledProcessError, ValueError, IndexError) as e:
            print(f"警告：无法从 Git 获取 {file_path} 的时间戳: {e}")

        return None

    def get_file_timestamp(self, file_path: Path) -> Tuple[datetime, datetime]:
        """从文件系统获取时间戳"""
        stat = file_path.stat()
        created = datetime.fromtimestamp(stat.st_ctime)
        modified = datetime.fromtimestamp(stat.st_mtime)
        return (created, modified)

    def extract_frontmatter(self, content: str) -> Tuple[Optional[dict], str]:
        """提取 YAML front matter"""
        frontmatter_pattern = re.compile(
            r'^---\s*\n(.*?)\n---\s*\n',
            re.DOTALL | re.MULTILINE
        )

        match = frontmatter_pattern.match(content)

        if match:
            frontmatter_text = match.group(1)
            rest_content = content[match.end():]

            # 解析 YAML（简单版本）
            frontmatter = {}
            for line in frontmatter_text.split('\n'):
                if ':' in line:
                    key, value = line.split(':', 1)
                    frontmatter[key.strip()] = value.strip()

            return (frontmatter, rest_content)

        return (None, content)

    def create_frontmatter(self, created: datetime, modified: datetime,
                          title: Optional[str] = None) -> str:
        """创建 YAML front matter"""
        lines = ['---']

        if title:
            lines.append(f'title: {title}')

        lines.extend([
            f'created: {created.strftime("%Y-%m-%d")}',
            f'updated: {modified.strftime("%Y-%m-%d")}',
            '---',
            ''
        ])

        return '\n'.join(lines)

    def update_frontmatter(self, frontmatter: dict, created: datetime,
                          modified: datetime) -> str:
        """更新现有的 front matter"""
        # 更新时间戳
        frontmatter['created'] = created.strftime("%Y-%m-%d")
        frontmatter['updated'] = modified.strftime("%Y-%m-%d")

        lines = ['---']
        for key, value in frontmatter.items():
            lines.append(f'{key}: {value}')
        lines.extend(['---', ''])

        return '\n'.join(lines)

    def extract_title(self, content: str) -> Optional[str]:
        """从内容中提取标题"""
        # 查找第一个 H1 标题
        match = re.search(r'^#\s+(.+)$', content, re.MULTILINE)
        if match:
            return match.group(1).strip()
        return None

    def update_file(self, file_path: Path) -> bool:
        """更新单个文件的时间戳"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            # 获取时间戳
            git_timestamp = self.get_git_timestamp(file_path)
            if git_timestamp:
                created, modified = git_timestamp
            else:
                created, modified = self.get_file_timestamp(file_path)

            # 提取 front matter
            frontmatter, rest_content = self.extract_frontmatter(content)

            # 提取标题
            title = self.extract_title(rest_content)

            # 生成新的 front matter
            if frontmatter:
                new_frontmatter = self.update_frontmatter(frontmatter, created, modified)
            else:
                new_frontmatter = self.create_frontmatter(created, modified, title)

            # 组合新内容
            new_content = new_frontmatter + rest_content

            # 只有内容发生变化时才写入
            if new_content != content:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(new_content)

                self.updated_files.append(file_path)
                return True
            else:
                self.skipped_files.append(file_path)
                return False

        except Exception as e:
            print(f"错误：更新 {file_path} 失败: {e}")
            return False

    def run(self) -> int:
        """运行时间戳更新"""
        print("🕐 开始更新时间戳...")
        print(f"📂 扫描目录: {self.root_dir}")

        if self.use_git:
            print("📝 使用 Git 历史记录")
        else:
            print("📝 使用文件系统时间戳")

        print()

        md_files = self.find_markdown_files()
        print(f"📄 找到 {len(md_files)} 个 Markdown 文件")
        print()

        for i, file_path in enumerate(md_files, 1):
            print(f"[{i}/{len(md_files)}] 处理: {file_path.relative_to(self.root_dir)}")
            self.update_file(file_path)

        print("\n" + "=" * 70)
        print("📊 更新结果:")
        print("=" * 70)
        print(f"✅ 更新文件: {len(self.updated_files)}")
        print(f"⏭️  跳过文件: {len(self.skipped_files)}")
        print()

        if self.updated_files:
            print("✅ 已更新的文件:")
            for file_path in self.updated_files[:10]:  # 只显示前10个
                print(f"  - {file_path.relative_to(self.root_dir)}")

            if len(self.updated_files) > 10:
                print(f"  ... 以及其他 {len(self.updated_files) - 10} 个文件")

            print()

        print("✨ 时间戳更新完成！")
        return 0


def main():
    import argparse

    parser = argparse.ArgumentParser(description='更新 Markdown 文件的时间戳')
    parser.add_argument(
        'directory',
        nargs='?',
        default='.',
        help='要处理的目录（默认：当前目录）'
    )
    parser.add_argument(
        '--no-git',
        action='store_true',
        help='不使用 Git 历史记录，使用文件系统时间戳'
    )

    args = parser.parse_args()

    root_dir = os.path.abspath(args.directory)

    if not os.path.isdir(root_dir):
        print(f"错误：{root_dir} 不是有效目录")
        return 1

    updater = TimestampUpdater(root_dir, use_git=not args.no_git)
    return updater.run()


if __name__ == '__main__':
    sys.exit(main())
