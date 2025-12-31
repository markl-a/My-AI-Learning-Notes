#!/usr/bin/env python3
"""
目录索引生成脚本
自动生成项目的目录结构和索引文件
"""

import os
import sys
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime
import re


class TOCGenerator:
    def __init__(self, root_dir: str):
        self.root_dir = Path(root_dir)
        self.tree_structure = []
        self.file_index = {}

        # 忽略的目录
        self.ignore_dirs = {
            '.git', '.github', 'node_modules', '__pycache__',
            '.vscode', '.idea', 'vendor', 'site', '.pytest_cache'
        }

        # 忽略的文件
        self.ignore_files = {
            '.DS_Store', 'Thumbs.db', '.gitignore', '.gitkeep'
        }

    def should_ignore(self, path: Path) -> bool:
        """判断是否应该忽略该路径"""
        # 检查是否在忽略列表中
        if path.name in self.ignore_dirs or path.name in self.ignore_files:
            return True

        # 检查是否以点开头（隐藏文件/目录）
        if path.name.startswith('.') and path.name not in {'.editorconfig'}:
            return True

        return False

    def extract_title_from_markdown(self, file_path: Path) -> Optional[str]:
        """从 Markdown 文件中提取标题"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    # 查找第一个 # 标题
                    match = re.match(r'^#\s+(.+)$', line.strip())
                    if match:
                        return match.group(1)

                    # 如果有 YAML front matter，跳过
                    if line.strip() == '---':
                        continue

            # 如果没有找到标题，使用文件名
            return file_path.stem.replace('-', ' ').replace('_', ' ').title()

        except Exception as e:
            print(f"警告：无法读取 {file_path}: {e}")
            return file_path.stem

    def get_file_info(self, file_path: Path) -> Dict:
        """获取文件信息"""
        stat = file_path.stat()

        info = {
            'path': str(file_path.relative_to(self.root_dir)),
            'name': file_path.name,
            'size': stat.st_size,
            'modified': datetime.fromtimestamp(stat.st_mtime),
        }

        if file_path.suffix == '.md':
            info['title'] = self.extract_title_from_markdown(file_path)
            info['type'] = 'markdown'
        else:
            info['title'] = file_path.name
            info['type'] = 'file'

        return info

    def build_tree(self, directory: Path, prefix: str = "", is_last: bool = True) -> List[str]:
        """构建目录树"""
        lines = []

        # 获取并排序所有项
        try:
            items = sorted(directory.iterdir(), key=lambda x: (not x.is_dir(), x.name))
        except PermissionError:
            return lines

        # 过滤掉应该忽略的项
        items = [item for item in items if not self.should_ignore(item)]

        for i, item in enumerate(items):
            is_last_item = (i == len(items) - 1)

            # 构建树形结构符号
            if is_last_item:
                connector = "└── "
                new_prefix = prefix + "    "
            else:
                connector = "├── "
                new_prefix = prefix + "│   "

            if item.is_dir():
                lines.append(f"{prefix}{connector}📁 **{item.name}/**")
                # 递归处理子目录
                lines.extend(self.build_tree(item, new_prefix, is_last_item))
            else:
                # 添加文件信息
                icon = self.get_file_icon(item)
                file_info = self.get_file_info(item)

                if item.suffix == '.md':
                    title = file_info.get('title', item.name)
                    rel_path = file_info['path']
                    lines.append(f"{prefix}{connector}{icon} [{title}]({rel_path})")
                else:
                    lines.append(f"{prefix}{connector}{icon} {item.name}")

                # 添加到文件索引
                self.file_index[file_info['path']] = file_info

        return lines

    def get_file_icon(self, file_path: Path) -> str:
        """根据文件类型返回图标"""
        ext = file_path.suffix.lower()

        icon_map = {
            '.md': '📄',
            '.py': '🐍',
            '.js': '📜',
            '.ts': '📘',
            '.json': '📋',
            '.yml': '⚙️',
            '.yaml': '⚙️',
            '.sh': '🔧',
            '.txt': '📝',
            '.pdf': '📕',
            '.png': '🖼️',
            '.jpg': '🖼️',
            '.jpeg': '🖼️',
            '.gif': '🖼️',
            '.svg': '🎨',
        }

        return icon_map.get(ext, '📄')

    def generate_category_index(self) -> Dict[str, List]:
        """按类别组织文件"""
        categories = {
            'Machine Learning': [],
            'Deep Learning': [],
            'NLP': [],
            'Computer Vision': [],
            'Tools & Frameworks': [],
            'Resources': [],
            'Other': []
        }

        for path, info in self.file_index.items():
            if info['type'] != 'markdown':
                continue

            path_lower = path.lower()

            if any(keyword in path_lower for keyword in ['machine-learning', 'ml', '机器学习']):
                categories['Machine Learning'].append(info)
            elif any(keyword in path_lower for keyword in ['deep-learning', 'dl', '深度学习']):
                categories['Deep Learning'].append(info)
            elif any(keyword in path_lower for keyword in ['nlp', 'natural-language', '自然语言']):
                categories['NLP'].append(info)
            elif any(keyword in path_lower for keyword in ['cv', 'computer-vision', '计算机视觉', 'vision']):
                categories['Computer Vision'].append(info)
            elif any(keyword in path_lower for keyword in ['tool', 'framework', '工具']):
                categories['Tools & Frameworks'].append(info)
            elif any(keyword in path_lower for keyword in ['resource', 'reference', '资源']):
                categories['Resources'].append(info)
            else:
                categories['Other'].append(info)

        return categories

    def generate_readme(self, output_path: Path):
        """生成 README.md 文件"""
        lines = [
            "# My AI Learning Notes",
            "",
            "全面的 AI 学习笔记和资源集合",
            "",
            f"📅 最后更新：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "## 📖 关于本项目",
            "",
            "这是一个个人 AI 学习笔记仓库，涵盖了机器学习、深度学习、自然语言处理、计算机视觉等多个领域的学习资源和实践经验。",
            "",
            "## 📚 内容分类",
            ""
        ]

        # 生成分类索引
        categories = self.generate_category_index()

        for category, files in categories.items():
            if not files:
                continue

            lines.append(f"### {category}")
            lines.append("")

            for file_info in sorted(files, key=lambda x: x['path']):
                title = file_info['title']
                path = file_info['path']
                lines.append(f"- [{title}]({path})")

            lines.append("")

        # 添加目录树
        lines.extend([
            "## 📂 目录结构",
            "",
            "```",
        ])

        tree_lines = self.build_tree(self.root_dir)
        lines.extend(tree_lines)

        lines.extend([
            "```",
            "",
            "## 🚀 快速开始",
            "",
            "### 浏览文档",
            "",
            "直接在 GitHub 上浏览，或访问 [在线文档网站](https://your-username.github.io/My-AI-Learning-Notes/)",
            "",
            "### 本地运行",
            "",
            "```bash",
            "# 克隆仓库",
            "git clone https://github.com/your-username/My-AI-Learning-Notes.git",
            "",
            "# 安装依赖",
            "pip install -r requirements.txt",
            "",
            "# 构建文档网站",
            "mkdocs serve",
            "```",
            "",
            "## 🤝 贡献",
            "",
            "欢迎贡献！请查看 [贡献指南](CONTRIBUTING.md) 了解详情。",
            "",
            "## 📜 许可证",
            "",
            "本项目采用 MIT 许可证 - 查看 [LICENSE](LICENSE) 文件了解详情。",
            "",
            "## 📧 联系方式",
            "",
            "如有问题或建议，请提出 [Issue](https://github.com/your-username/My-AI-Learning-Notes/issues)。",
            "",
            "---",
            "",
            f"⭐ 如果这个项目对你有帮助，请给个 Star！",
        ])

        # 写入文件
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))

        print(f"✅ 生成 README: {output_path}")

    def generate_index(self, output_path: Path):
        """生成 index.md 文件（用于 MkDocs）"""
        lines = [
            "# 欢迎来到 My AI Learning Notes",
            "",
            "这是一个系统化的 AI 学习笔记集合。",
            "",
            "## 最近更新",
            ""
        ]

        # 获取最近修改的文件
        recent_files = sorted(
            [info for info in self.file_index.values() if info['type'] == 'markdown'],
            key=lambda x: x['modified'],
            reverse=True
        )[:10]

        for file_info in recent_files:
            modified = file_info['modified'].strftime('%Y-%m-%d')
            title = file_info['title']
            path = file_info['path']
            lines.append(f"- **{modified}** - [{title}]({path})")

        lines.extend([
            "",
            "## 学习路径",
            "",
            "推荐按以下顺序学习：",
            "",
            "1. 基础概念",
            "2. 机器学习",
            "3. 深度学习",
            "4. 专业领域（NLP、CV 等）",
            "",
            "## 统计信息",
            "",
            f"- 📄 Markdown 文件数：{len([f for f in self.file_index.values() if f['type'] == 'markdown'])}",
            f"- 📁 总文件数：{len(self.file_index)}",
            f"- 📅 最后更新：{datetime.now().strftime('%Y-%m-%d')}",
        ])

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))

        print(f"✅ 生成 index.md: {output_path}")

    def run(self):
        """执行目录生成"""
        print("🔨 开始生成目录索引...")
        print(f"📂 根目录: {self.root_dir}")
        print()

        # 构建文件索引
        self.build_tree(self.root_dir)

        # 生成 README.md
        readme_path = self.root_dir / "README.md"
        self.generate_readme(readme_path)

        # 生成 index.md（用于文档网站）
        index_path = self.root_dir / "index.md"
        self.generate_index(index_path)

        print()
        print(f"✨ 完成！共索引 {len(self.file_index)} 个文件")


def main():
    import argparse

    parser = argparse.ArgumentParser(description='生成项目目录索引')
    parser.add_argument(
        'directory',
        nargs='?',
        default='.',
        help='项目根目录（默认：当前目录）'
    )

    args = parser.parse_args()

    root_dir = os.path.abspath(args.directory)

    if not os.path.isdir(root_dir):
        print(f"错误：{root_dir} 不是有效目录")
        return 1

    generator = TOCGenerator(root_dir)
    generator.run()

    return 0


if __name__ == '__main__':
    sys.exit(main())
