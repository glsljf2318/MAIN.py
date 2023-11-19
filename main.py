import random
import matplotlib.pyplot as plt
import numpy as np
import mpl_toolkits.axisartist as axisartist
import math as m


def Heavyside(x):
    if x < 0:
        return (0)
    else:
        return (1)


def Eiler(x, dx):
    h = 0.001
    x = x + np.dot(dx, h)
    return x


def Intensity(x):
    xu = np.sqrt(2 / 3) * np.sqrt(np.tensordot(x, x))
    return xu


class Monocristall:

    def __init__(self):
        self.P = np.array([[[[0] * 3 for i in range(3)] for i in range(3)] for i in range(3)],
                          dtype=np.float64)  # Тензор упругих свойств
        self.P[0, 0, 0, 0] = self.P[1, 1, 1, 1] = self.P[2, 2, 2, 2] = 107300000000
        self.P[0, 0, 1, 1] = self.P[1, 1, 0, 0] = self.P[2, 2, 1, 1] = 60900000000
        self.P[1, 1, 2, 2] = self.P[2, 2, 0, 0] = self.P[0, 0, 2, 2] = 60900000000
        self.P[0, 1, 1, 0] = self.P[1, 2, 1, 2] = self.P[0, 1, 0, 1] = self.P[2, 0, 2, 0] = 28300000000
        self.P[0, 2, 0, 2] = self.P[2, 1, 2, 1] = self.P[1, 0, 1, 0] = self.P[1, 0, 0, 1] = 28300000000
        self.P[2, 1, 1, 2] = self.P[0, 2, 2, 0] = self.P[1, 2, 2, 1] = self.P[2, 0, 0, 2] = 28300000000
        self.d = np.array([[0, 0.001, 0], [0, 0, 0], [0, 0, 0]])  # d elastic для начальной итер для простого сдвига
        self.a = np.zeros((12, 12), dtype=np.float64)  # матрица a
        self.din = np.zeros((3, 3), dtype=np.float64)
        self.Eiler_angles = np.random.uniform(0, 360, 3)
        for n in range(12):
            for t in range(12):
                if n == t:
                    self.a[n][t] = 4.56e-5
                else:
                    self.a[n][t] = 5.7e-5
        with open('normaly', 'r') as f1:
            lst = f1.readlines()
        self.table1 = [s.split() for s in lst]
        with open('vectors_b', 'r') as f2:
            lst = f2.readlines()
        self.table2 = [s.split() for s in lst]

    def zn_Huka(self, din, O_matrix):
        P = self.P
        d = np.dot(np.dot(O_matrix, self.d), O_matrix.T)

        dG = np.ones((3, 3), dtype=np.float64)  # производная от тензора напряжений
        dG = np.tensordot(P, (d - din))
        return dG

    def Rotation_matrix(self):
        E_a = self.Eiler_angles
        Prec_matrix = np.array([[m.cos(m.radians(E_a[0])), (-1) * m.sin(m.radians(E_a[0])), 0.],
                                [m.sin(m.radians(E_a[0])), m.cos(m.radians(E_a[0])), 0.], [0., 0., 1.]],
                               dtype=np.float64)
        Nutat_matrix = np.array([[1., 0., 0.], [0., m.cos(m.radians(E_a[1])), (-1) * m.sin(m.radians(E_a[1]))],
                                 [0., m.sin(m.radians(E_a[1])), m.cos(m.radians(E_a[1]))]], dtype=np.float64)
        Own_rot_matrix = np.array([[m.cos(m.radians(E_a[2])), (-1) * m.sin(m.radians(E_a[2])), 0.],
                                   [m.sin(m.radians(E_a[2])), m.cos(m.radians(E_a[2])), 0.], [0., 0., 1.]],
                                  dtype=np.float64)
        O_matrix = np.dot(np.dot(Prec_matrix, Nutat_matrix), Own_rot_matrix)
        return O_matrix

    def OrienTensor(self, massiv_M):
        for i in range(12):
            b = self.table2[i]
            n = self.table1[i]
            M = np.zeros((3, 3), dtype=np.float64)  # ориентационный тензор
            for j in range(3):
                for k in range(3):
                    M[j, k] = (float(b[j]) * float(n[k]))
            massiv_M.append(M)
        return massiv_M

    def d_inelastic(self, massiv_M, G, tc_list, gamma_list):
        gamma0 = 10e-9
        m = 90
        din = self.din
        for i, M in enumerate(massiv_M):
            MT = M.T
            tk = np.tensordot(G, MT)
            gamma_k = gamma0 * pow(abs(tk / tc_list[i]), m) * Heavyside(abs(tk) - tc_list[i]) * np.sign(tk)
            din = din + np.dot(gamma_k, M)
            gamma_list[i] = gamma_k
        return din, gamma_list

    def DTc(self, gamma_list, tc_list):
        E = 70e9
        a = self.a
        dtc_list = E * np.dot(a, gamma_list)
        tc_list = Eiler(tc_list, dtc_list)
        return tc_list

    def Averaging(self, tuple):
        tuple = np.array(tuple).T
        massiv_mid = [0] * len(tuple)
        for count, i in enumerate(tuple):
            for j in i:
                massiv_mid[count] += (j ** 2)
            massiv_mid[count] = ((massiv_mid[count] / len(i)) ** (1 / 2))
        return massiv_mid

    def Draw_Poly(self, tuple_Gu, tuple_Eu, data):
        fig = plt.figure()
        colors = ["black", "blue", "red", "orange", "green", "pink", "yellow", "brown"]
        # Используйте axisartist.Subplot метод для создания области рисования объекта ax
        ax = axisartist.Subplot(fig, 111)
        # Добавить объекты области рисования на холст
        fig.add_axes(ax)
        # Установите стили нижней и левой оси области рисования с помощью метода set_axisline_style
        # "- |>" представляет сплошную стрелку: "->" представляет полую стрелку
        ax.axis["bottom"].set_axisline_style("-|>", size=1.5)
        ax.axis["left"].set_axisline_style("-|>", size=1.5)
        for i in range(len(data) ** 3):
            if i % 2 == 0:
                plt.plot(tuple_Eu[i], tuple_Gu[i], color=colors[0])
            else:
                plt.plot(tuple_Eu[i], tuple_Gu[i], color=colors[1])
        plt.ylim(bottom=0, top=max(max(tuple_Gu)) + 0.5e7)
        plt.xlim(left=0, right=max(max(tuple_Eu)))
        plt.xlabel("ε")
        plt.ylabel("σ")
        plt.show()

    def Draw_Mono(self, massive_Gu, massiv_Eu):
        fig = plt.figure()
        # Используйте axisartist.Subplot метод для создания области рисования объекта ax
        ax = axisartist.Subplot(fig, 111)
        # Добавить объекты области рисования на холст
        fig.add_axes(ax)
        # Установите стили нижней и левой оси области рисования с помощью метода set_axisline_style
        # "- |>" представляет сплошную стрелку: "->" представляет полую стрелку
        ax.axis["bottom"].set_axisline_style("-|>", size=1.5)
        ax.axis["left"].set_axisline_style("-|>", size=1.5)
        plt.plot(massiv_Eu, massive_Gu, color="black")
        plt.ylim(bottom=0, top=max(massive_Gu))
        plt.xlim(left=0)
        plt.xlabel("ε")
        plt.ylabel("σ")
        plt.show()

    def Draw_3D(self, data):
        def explode(data):
            size = np.array(data.shape) * 2
            data_e = np.zeros(size - 1, dtype=data.dtype)
            data_e[::2, ::2, ::2] = data
            return data_e

        # build up the numpy logo
        n_voxels = np.zeros((len(data), len(data), len(data)), dtype=np.float64)
        for i in range(np.shape(n_voxels)[0]):
            for j in range(np.shape(n_voxels)[0]):
                for k in range(np.shape(n_voxels)[0]):
                    a, b, c = data[i, j, k]
                    n_voxels[i, j, k] = a
        n_voxels = n_voxels - np.max(n_voxels)
        n_voxels = abs(n_voxels / np.min(n_voxels))
        facecolors = np.zeros((len(data), len(data), len(data)), dtype=object)
        for i in range(np.shape(n_voxels)[0]):
            for j in range(np.shape(n_voxels)[0]):
                for k in range(np.shape(n_voxels)[0]):
                    facecolors[i, j, k] = (0.9, 0.0, 0.0, n_voxels[i, j, k])
                    """(0.9, 0.0, 0.0, n_voxels[i, j, k])"""
        filled = np.ones(n_voxels.shape)

        # upscale the above voxel image, leaving gaps
        filled_2 = explode(filled)
        fcolors_2 = explode(facecolors)

        # Shrink the gaps
        x, y, z = np.indices(np.array(filled_2.shape) + 1).astype(float) // 2
        x[0::2, :, :] += 0.05
        y[:, 0::2, :] += 0.05
        z[:, :, 0::2] += 0.05
        x[1::2, :, :] += 0.95
        y[:, 1::2, :] += 0.95
        z[:, :, 1::2] += 0.95

        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.voxels(x, y, z, filled_2, facecolors=fcolors_2, edgecolors=fcolors_2)
        ax.set_xlabel('X-axis')
        ax.set_ylabel('Y-axis')
        ax.set_zlabel('Z-axis')
        plt.show()


"""ОСНОВНАЯ ПРОГРАММА(ЦИКЛ ДЛЯ РАСЧЕТА НДС)"""
data = np.zeros((7, 7, 7), dtype=object)
tuple_Gu = []
tuple_Eu = []
for i in range(len(data)):
    for j in range(len(data)):
        for k in range(len(data)):
            tc_list = [17500000.] * 12
            gamma_list = [1.] * 12
            massiv_Gu = []  # все значения G интесив
            massiv_Eu = []  # все значения E интенсив
            massiv_M = []
            G = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]])  # ж ноликовое
            Eps = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]])  # Eps ноликовое
            din = np.zeros((3, 3), dtype=np.float64)
            Al = Monocristall()
            Al.OrienTensor(massiv_M)
            O_matrix = Al.Rotation_matrix()
            znak = random.choice([-100, 100])
            Eu = 0
            Gu = 0
            count = 0
            # din = np.zeros((3, 3), dtype=np.float64)
            while (Eu < 0.1):
                count += 1
                G = Eiler(G, Al.zn_Huka(din, O_matrix))
                d = np.dot(np.dot(O_matrix, Al.d), O_matrix.T)
                Eps = Eiler(Eps, d)
                Gu = Intensity(G)
                Eu = Intensity(Eps)
                massiv_Gu.append(Gu)
                massiv_Eu.append(Eu)
                din, gamma_list = Al.d_inelastic(massiv_M, G, tc_list, gamma_list)
                tc_list = Al.DTc(gamma_list, tc_list)
            tuple_Gu.append(massiv_Gu)
            tuple_Eu.append(massiv_Eu)
            print("#############")
            data[i, j, k] = (max(massiv_Gu), (i, j, k), znak)
Al = Monocristall()
Al.Draw_Poly(tuple_Gu, tuple_Eu, data)


"""ОСРЕДНЕНИЕ ПО ВСЕМ КРИСТАЛЛИТАМ"""
Al.Draw_Mono(Al.Averaging(tuple_Gu), Al.Averaging(tuple_Eu))


"""ОТРИСОВКА КУБИКА"""
Al.Draw_3D(data)



