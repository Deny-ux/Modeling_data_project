fun = @(x) x^2 - 4;
x0 = 1;
[x, fval, exitflag, output] = fsolve(fun, x0);
disp(x);