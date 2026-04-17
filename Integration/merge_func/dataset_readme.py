import os


class ReadMeElement:
    def __init__(self, text, level=0, node_type='content'):
        self.text = text
        self.level = level
        self.node_type = node_type
        self.children = []
        self.meta = {}

    def add_child(self, child_node):
        self.children.append(child_node)
        return child_node


class ReadMeBuilder:
    def __init__(self):
        self.root = ReadMeElement(text="Project Name", level=1, node_type='header')
        # Initialize stack with root to ensure there is always a base context

    def add_header(self, text, level=2):
        """Add a header without entering the hierarchy (Flat structure)"""
        node = ReadMeElement(text=text, level=level, node_type='header')
        self.root.add_child(node)
        return self

    def add_text(self, text):
        node = ReadMeElement(text=text, level=0, node_type='text')
        self.root.add_child(node)
        return self

    def add_code(self, code_str, language="python"):
        node = ReadMeElement(text=code_str, level=0, node_type='code')
        node.meta['language'] = language
        self.root.add_child(node)
        return self

    def add_list(self, items, ordered=False):
        node = ReadMeElement(text="", level=0, node_type='list')
        node.meta['items'] = items
        node.meta['ordered'] = ordered
        self.root.add_child(node)
        return self

    def _render_node(self, node):
        md_lines = []

        if node.node_type == 'header':
            prefix = '#' * node.level
            md_lines.append(f"{prefix} {node.text}\n")
        elif node.node_type == 'text':
            md_lines.append(f"{node.text}\n")
        elif node.node_type == 'code':
            lang = node.meta.get('language', '')
            md_lines.append(f"```{lang}\n{node.text}\n```\n")
        elif node.node_type == 'list':
            items = node.meta.get('items', [])
            is_ordered = node.meta.get('ordered', False)
            for i, item in enumerate(items):
                marker = f"{i + 1}." if is_ordered else "-"
                md_lines.append(f"{marker} {item}\n")

        md_lines.append("\n")

        for child in node.children:
            md_lines.append(self._render_node(child))

        return "".join(md_lines)

    def generate(self, path):
        try:
            content = self._render_node(self.root)
            with open(path, "w", encoding="utf-8") as f:
                f.write(content)
            # Success: File generated
            print(f"[✅ SUCCESS] README.md generated at: {os.path.abspath(path)}")
        except Exception as e:
            # Error: File generation failed
            print(f"[❌ ERROR] Failed to generate file: {str(e)}")