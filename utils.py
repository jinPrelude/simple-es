def slice_list(lst: list, n: int) -> list:
    # 리스트를 n개만큼 자름
    result = []
    size = int(len(lst) / n)
    for i in range(0, n):
        result.append(lst[i * size : (i + 1) * size])
    if size * n < len(lst):
        j = size * n
        k = 0
        # assign extra data
        while j < len(lst):
            result[k].append(lst[j])
            j += 1
            k += 1
            if k >= len(result):
                k = 0
    return result
