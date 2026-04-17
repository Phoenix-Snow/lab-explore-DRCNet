import Integration.Private_Dataset_Solar.Merge as Merge

if __name__ == '__main__':
    back = Merge.MergeByYaml().work()
    Merge.MergeByYaml().not_work(back[0])