from sympy import Matrix, sin, pi, cos, symbols, sqrt, atan

t, tm, b = symbols('t tm b')


class BaseArrayConfiguration:
    acm = None


class XArrayConfiguration(BaseArrayConfiguration):
    q = 6
    acm = (b / 2
           * Matrix([[cos(2 * pi / tm * t), -sin(2 * pi / tm * t)],
                     [sin(2 * pi / tm * t), cos(2 * pi / tm * t)]])
           * Matrix([[q, q, -q, -q],
                     [1, -1, -1, 1]]))


class KiteArrayConfiguration(BaseArrayConfiguration):
    c = 1.69
    beta = 2 * atan(1 / c)
    th = [pi / 2 - beta, pi / 2, pi / 2 + beta, 3 * pi / 2]

    acm = (b / 2 * sqrt(1 + c ** 2)
           * Matrix([[cos(2 * pi / tm * t), -sin(2 * pi / tm * t)],
                     [sin(2 * pi / tm * t), cos(2 * pi / tm * t)]])
           * Matrix([[cos(th[0]), cos(th[1]), cos(th[2]), cos(th[3])],
                     [sin(th[0]), sin(th[1]), sin(th[2]), sin(th[3])]]))


class LinearArrayConfiguration(BaseArrayConfiguration):
    c = 1.69
    beta = 2 * atan(1 / c)
    th = [pi / 2 - beta, pi / 2, 2 * pi / 2, 3 * pi / 2]
    q = 1
    acm = (b / 2
           * Matrix([[cos(2 * pi / tm * t), -sin(2 * pi / tm * t)],
                     [sin(2 * pi / tm * t), cos(2 * pi / tm * t)]])
           * Matrix([[2, 1, -1.5, -3],
                     [0, 0, 0, 0]]))


class PentagonArrayConfiguration(BaseArrayConfiguration):
    th = [0, 2 * pi / 5, 4 * pi / 5, 6 * pi / 5, 8 * pi / 5]
    acm = (0.851 * b
           * Matrix([[cos(2 * pi / tm * t), -sin(2 * pi / tm * t)],
                     [sin(2 * pi / tm * t), cos(2 * pi / tm * t)]])
           * Matrix([[cos(th[0]), cos(th[1]), cos(th[2]), cos(th[3]), cos(th[4])],
                     [sin(th[0]), sin(th[1]), sin(th[2]), sin(th[3]), sin(th[4])]]))


class Test(BaseArrayConfiguration):
    th = [0.28689979, 1.22718398, 2.1642926, 2.35099541, 3.72472115]
    fb = [2.11430767, 1.02492582, 1.57638794, 2.55903274, 1.25833183]
    fb1, fb2, fb3, fb4, fb5 = fb
    acm = (0.851 * b
           * Matrix([[cos(2 * pi / tm * t), -sin(2 * pi / tm * t)],
                     [sin(2 * pi / tm * t), cos(2 * pi / tm * t)]])
           * Matrix([[fb1 * cos(th[0]), fb2 * cos(th[1]), fb3 * cos(th[2]), fb4 * cos(th[3]), fb5 * cos(th[4])],
                     [fb1 * sin(th[0]), fb2 * sin(th[1]), fb3 * sin(th[2]), fb4 * sin(th[3]), fb5 * sin(th[4])]]))
