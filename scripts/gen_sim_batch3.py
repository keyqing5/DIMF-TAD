import numpy as np
from scipy.special import comb
import random
import argparse
import seaborn as sns
import matplotlib.pyplot as plt
import os


class Simulator():
    # 移除了 max_n 和 min_n
    def __init__(self, matrix_size, height, output, max_size, min_size, over_up, over_down, gap, noise):
        self.Number = matrix_size
        self.height = height
        self.noise_ratio = noise
        self.overlap_ratio_up = over_up
        self.overlap_ratio_down = over_down
        self.background_ratio = 0.5 * self.noise_ratio / (1 - self.noise_ratio)
        self.edge_ratio = 1
        self.output = output
        self.gap = gap
        self.gap_TAD = [-1, -1]

        self.tad = []
        current_sum = 0
        while current_sum < self.Number:

            s = random.randint(int(min_size), int(max_size))

            if current_sum + s > self.Number:
                s = self.Number - current_sum

            remaining = self.Number - (current_sum + s)
            if 0 < remaining < min_size:
                s += remaining

            self.tad.append(s)
            current_sum += s
            if current_sum >= self.Number:
                break

        self.TAD_n = len(self.tad)
        self.list_boundary = []
        init_pos = 1
        boundary = [init_pos]
        for i in range(0, len(self.tad), 1):
            init_pos += self.tad[i]
            boundary.append(init_pos)

        self.matrix = None
        self.boundary = boundary

    def generate(self):
        self.matrix = self.add_background()
        self.edge_ratio = self.edge_ratio - 0.2
        if self.overlap_ratio_up > self.overlap_ratio_down:
            self.overlap_boundary = []
            self.overlap_boundary.append([self.boundary[0], self.boundary[1] - 1])
            for i in range(2, len(self.boundary), 1):
                left_boundary = int((self.boundary[i - 1] - self.boundary[i - 2]) * \
                                    random.uniform(self.overlap_ratio_down, self.overlap_ratio_up) + \
                                    self.boundary[i - 2])
                self.overlap_boundary.append([left_boundary - 1, self.boundary[i] - 1])

            if self.gap:
                random_index = random.randrange(1, len(self.overlap_boundary) - 2)
                self.gap_TAD[0] = self.overlap_boundary[random_index - 1][1] + 1
                self.gap_TAD[1] = self.overlap_boundary[random_index + 1][0] - 1

            self.matrix += self.generate_overlap()
            self.list_boundary = self.list_boundary + self.overlap_boundary
        else:
            self.non_overlap_boundary = []
            for i in range(0, len(self.boundary) - 1, 1):
                self.non_overlap_boundary.append([self.boundary[i], self.boundary[i + 1] - 1])

            if self.gap:
                random_index = random.randrange(1, len(self.non_overlap_boundary) - 2)
                self.gap_TAD = self.non_overlap_boundary[random_index]
            self.matrix += self.generate_non_overlap()
            self.list_boundary = self.list_boundary + self.non_overlap_boundary

        self.height = self.height - 1
        for h in range(self.height, 0, -1):
            high_level_boundary = []
            self.edge_ratio -= 0.1
            temp_boundary = self.boundary[::3]  # 模拟高层级 TAD
            for i in range(0, len(temp_boundary) - 1, 1):
                high_level_boundary.append([temp_boundary[i], temp_boundary[i + 1] - 1])

            # 使用 temp_boundary 生成高层矩阵
            temp_matrix = np.zeros((self.Number, self.Number))
            for i in range(0, len(temp_boundary) - 1, 1):
                TAD_size = temp_boundary[i + 1] - temp_boundary[i]
                if TAD_size < 2: continue
                edge_num = int((comb(TAD_size, 2) - TAD_size + 1) * self.edge_ratio)
                in_list = []
                for j in range(temp_boundary[i], temp_boundary[i + 1] - 2):
                    for k in range(j + 2, temp_boundary[i + 1]):
                        in_list.append([j - 1, k - 1])
                if in_list:
                    sample_n = min(len(in_list), edge_num)
                    in_coordinate = random.sample(in_list, sample_n)
                    for j in range(0, sample_n):
                        temp_matrix[in_coordinate[j][0], in_coordinate[j][1]] = self.height * 100
            self.matrix += temp_matrix
            self.list_boundary = self.list_boundary + high_level_boundary

        for i in range(0, self.Number - 1):
            self.matrix[i, i] = 0
            self.matrix[i, i + 1] = self.matrix.max()
        self.matrix = np.triu(self.matrix)
        self.matrix += self.matrix.T - np.diag(self.matrix.diagonal())

        if self.gap:
            for i in range(max(0, self.gap_TAD[0] - 1), min(self.Number, self.gap_TAD[1])):
                self.matrix[i, :] = 0
                self.matrix[:, i] = 0
            self.list_boundary = [b for b in self.list_boundary if
                                  not (b[1] >= self.gap_TAD[1] and b[0] <= self.gap_TAD[0])]

        return self.list_boundary, self.matrix

    def generate_non_overlap(self):
        matrix = np.zeros((self.Number, self.Number))
        for i in range(0, len(self.boundary) - 1, 1):
            TAD_size = self.boundary[i + 1] - self.boundary[i]
            if self.boundary[i + 1] >= self.gap_TAD[1] and self.boundary[i] <= self.gap_TAD[0]:
                continue
            if TAD_size < 2: continue
            edge_num = int((comb(TAD_size, 2) - TAD_size + 1) * self.edge_ratio)
            in_list = []
            for j in range(self.boundary[i], self.boundary[i + 1] - 2):
                for k in range(j + 2, self.boundary[i + 1]):
                    in_list.append([j - 1, k - 1])
            if in_list:
                sample_n = min(len(in_list), edge_num)
                in_coordinate = random.sample(in_list, sample_n)
                for j in range(0, sample_n):
                    matrix[in_coordinate[j][0], in_coordinate[j][1]] = self.height * 100
        return matrix

    def generate_overlap(self):
        bounday_list = self.overlap_boundary
        matrix = np.zeros((self.Number, self.Number))
        for TAD in bounday_list:
            TAD_size = TAD[1] - TAD[0]
            if TAD[1] >= self.gap_TAD[1] and TAD[0] <= self.gap_TAD[0]:
                continue
            if TAD_size < 2: continue
            edge_num = int((comb(TAD_size, 2) - TAD_size + 1) * self.edge_ratio)
            in_list = []
            for j in range(TAD[0], TAD[1] - 2):
                for k in range(j + 2, TAD[1] + 1):
                    in_list.append([j - 1, k - 1])
            if in_list:
                sample_n = min(len(in_list), edge_num)
                in_coordinate = random.sample(in_list, sample_n)
                for j in range(0, sample_n):
                    matrix[in_coordinate[j][0], in_coordinate[j][1]] += (self.height + 1) * 100 + 100
        return matrix

    def add_background(self):
        matrix = np.zeros((self.Number, self.Number))
        in_list_total = []
        for j in range(1, self.Number - 2):
            for k in range(j + 2, self.Number):
                in_list_total.append([j - 1, k - 1])
        edge_num_total = int((comb(self.Number, 2) - self.Number + 1) * self.background_ratio)
        if in_list_total:
            sample_n = min(len(in_list_total), edge_num_total)
            in_coordinate = random.sample(in_list_total, sample_n)
            for j in range(0, sample_n):
                matrix[in_coordinate[j][0], in_coordinate[j][1]] = 100
        return matrix


def bound2domain(bound, filename, key_word="sim"):
    with open(filename, "w") as file:
        for b in bound:
            line = f"{key_word} {int(b[0])} 0 0 {key_word} {int(b[1])}\n"
            file.write(line)


def main():
    parser = argparse.ArgumentParser(description="Simulate Matrix with Fixed Size")
    parser.add_argument("--matrix_size", dest="matrix_size", type=int, default=100,
                        help="Fixed matrix size (e.g. 100 for 100x100)")
    parser.add_argument("--height", dest="height", type=int, default=1, help="max height of the TAD")
    parser.add_argument("--max_size", dest="max_size", type=int, default=20, help="max size for each TAD")
    parser.add_argument("--min_size", dest="min_size", type=int, default=10, help="min size for each TAD")
    parser.add_argument("--over_up", dest="over_up", type=float, default=0.0, help="max overlap ratio")
    parser.add_argument("--over_down", dest="over_down", type=float, default=0.0, help="min overlap ratio")
    parser.add_argument("--gap", dest="gap", type=str, default="false", help="enable gap (true/false)")
    parser.add_argument("--noise", dest="noise", type=float, default=0.2, help="noise ratio")
    parser.add_argument("--out_dir", dest="out_dir", type=str, required=True, help="Output directory")
    parser.add_argument("--num", dest="num", type=int, default=10, help="Number of samples")

    args = parser.parse_args()
    gap_bool = args.gap.lower() == 'true'

    folder_name = f"M{args.matrix_size}_h{args.height}_s{args.min_size}_S{args.max_size}_o{args.over_down}_O{args.over_up}_n{args.noise}"
    full_target_dir = os.path.join(args.out_dir, folder_name)
    matrix_dir = os.path.join(full_target_dir, "matrix")
    gt_dir = os.path.join(full_target_dir, "gt")

    os.makedirs(matrix_dir, exist_ok=True)
    os.makedirs(gt_dir, exist_ok=True)

    for idx in range(args.num):
        sim = Simulator(matrix_size=args.matrix_size, height=args.height, output=None,
                        max_size=args.max_size, min_size=args.min_size,
                        over_up=args.over_up, over_down=args.over_down,
                        gap=gap_bool, noise=args.noise)

        real_boundary, simu_matrix = sim.generate()

        file_base_name = f"{folder_name}_T{idx}"
        np.savetxt(os.path.join(matrix_dir, f"{file_base_name}.txt"), simu_matrix, fmt='%.4f', delimiter='\t')
        bound2domain(bound=real_boundary, filename=os.path.join(gt_dir, f"{file_base_name}.domains"))

    print(f"Done! Generated {args.num} samples of size {args.matrix_size} in {full_target_dir}")


if __name__ == '__main__':
    main()