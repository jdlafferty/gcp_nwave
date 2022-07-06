# matrix
def print_matrix(m):
    for row in m:
        print(row)

def transpose(m):
    return [[m[j][i] for j in range(len(m))] for i in range(len(m[0]))]

def multiply(a, b):
    ra, ca = len(a), len(a[0])
    rb, cb = len(b), len(b[0])

    output = [[0 for _ in range(cb)] for _ in range(ra)]
    for i in range(ra):
        for j in range(cb):
            for k in range(rb):
                output[i][j] += a[i][k] * b[k][j]
    return output

# vector
def dot(a, b):
    sum = 0
    for i in range(len(a)):
        sum += a[i] * b[i]
    return sum

def norm(v):
    return (dot(v, v))**(1/2)

def right_multiply(m, v):
    output = []
    for row in m:
        output.append(dot(v, row))
    return output

def left_multiply(v, m):
    output = [0] * len(m[0])
    for i in range(len(m)):
        for j in range(len(m[0])):
            output[j] += v[i] * m[i][j]
    return output


if __name__ == "__main__":

    m = [[1,2],[3,4],[5,6]]
    v = [1,1]
    print_matrix(m)
    print_matrix(multiply(transpose(m), m))
    print(right_multiply(m, v))
