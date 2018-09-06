# -*- coding: utf-8 -*-
from IPython.core.magic import (Magics, magics_class, cell_magic, line_magic, line_cell_magic)
import os
import codecs

@magics_class
class MLCodeMagic(Magics):
    """ jupyter に書いた任意のセルをコードにするマジック
    - %%mlcodes でコードを記述。run_local と続けて記載するとローカルのjupyterでも同じコードを動かす。
      - 重たい学習で、ローカルで動かしたくないときは run_local を書かない
    - %code_to_pyfile でコードを ./trainer/task.py に保存。これを行うまではコードは保存されない。
    - %clear_mlcode でコードの初期化。%%mlcodes のセルをたくさん叩いちゃったときに直す用。
    """
    def __init__(self, shell=None):
        super(MLCodeMagic, self).__init__(shell=shell)
        self.mlcode = []


    @cell_magic
    def mlcodes(self, line, cell):
        self.mlcode.append(cell + '\n')

        # run_local を入れるとローカルのjupyterでも動かす
        run_local = line.strip() == 'run_local'
        if run_local:
            self.shell.run_cell(cell)
        return


    @line_magic
    def code_to_pyfile(self, line):
        if len(self.mlcode) == 0:
            raise BaseException('No ML code. Write any ML code with using mlcodes magic.')
            return

        # タスク格納先を作成し、コードを保存
        task_file = './trainer/task.py'
        task_file_path = os.path.dirname(task_file)
        if not os.path.exists(task_file_path):
            os.mkdir(task_file_path)

        with codecs.open('./trainer/__init__.py','w','utf-8') as f:
            f.write("")
        with codecs.open('./trainer/task.py','w','utf-8') as f:
            for r in self.mlcode:
                f.write(r)

    @line_magic
    def clear_mlcode(self, line):
        self.mlcode = []
