import numpy as np

import torch
from torch.autograd import Variable
from utils import *

def conjugate_gradients(Avp, b, nsteps, residual_tol=1e-10):
    x = torch.zeros(b.size())
    r = b.clone()
    p = b.clone()
    rdotr = torch.dot(r, r)
    for i in range(nsteps):
        _Avp = Avp(p)
        alpha = rdotr / torch.dot(p, _Avp)
        x += alpha * p
        r -= alpha * _Avp
        new_rdotr = torch.dot(r, r)
        betta = new_rdotr / rdotr
        p = r + betta * p
        rdotr = new_rdotr
        if rdotr < residual_tol:
            break
    return x


def linesearch(model,
               f,
               x,
               fullstep,
               expected_improve_rate,
               max_backtracks=10,
               accept_ratio=.1):
    fval = f(True).data
    print("fval before", fval[0])
    for (_n_backtracks, stepfrac) in enumerate(.5**np.arange(max_backtracks)):
        xnew = x + stepfrac * fullstep
        set_flat_params_to(model, xnew)
        newfval = f(True).data
        actual_improve = fval - newfval
        expected_improve = expected_improve_rate * stepfrac
        ratio = actual_improve / expected_improve
        print("a/e/r", actual_improve[0], expected_improve[0], ratio[0])

        if ratio[0] > accept_ratio and actual_improve[0] > 0:
            print("fval after", newfval[0])
            return True, xnew
    return False, x

def aggregate_or_eval_grads_qae(model, returns_arr, losses, num_eval_grad_steps, grads_list, args, writer, descs):
    variances = []
    get_oracle_eval_loss = losses[0]
    get_qe_loss = losses[1]
    get_qae_loss = losses[1]
    for desc, return_arr in zip(descs, returns_arr):
        if num_eval_grad_steps % args.eval_grad_freq == 0:
            total_grads = np.vstack(grads_list[descs.index(desc)])
            variance = np.log(np.mean(np.var(total_grads, 0)))
            print("Log Gradient Variance for {} model: {}".format(desc, variance))
            variances.append(variance)
            grads_list[descs.index(desc)] = []
        else:
            if desc[0] != 'q':
                eval_loss = get_oracle_eval_loss(return_arr)
            elif desc == 'qvalue':
                eval_loss = get_qe_loss(return_arr)[0]
            elif desc == 'qevalue':
                eval_loss = get_qae_loss(return_arr)[0]
            grads = torch.autograd.grad(eval_loss, model.parameters())
            loss_grad = torch.cat([grad.view(-1) for grad in grads]).data
            grads_list[descs.index(desc)].append(loss_grad.numpy())

    if len(variances) == len(descs):
        for var in variances:
            writer.write("{},".format(var))
        writer.write("\n")

    return grads_list

def aggregate_or_eval_grads_qe(model, returns_arr, losses, num_eval_grad_steps, grads_list, args, writer, descs):
    variances = []
    get_oracle_eval_loss = losses[0]
    get_qe_loss = losses[1]
    for desc, return_arr in zip(descs, returns_arr):
        if num_eval_grad_steps % args.eval_grad_freq == 0:
            total_grads = np.vstack(grads_list[descs.index(desc)])
            variance = np.log(np.mean(np.var(total_grads, 0)))
            print("Log Gradient Variance for {} model: {}".format(desc, variance))
            variances.append(variance)
            grads_list[descs.index(desc)] = []
        else:
            if desc[0] != 'q':
                eval_loss = get_oracle_eval_loss(return_arr)
            elif desc == 'qvalue':
                eval_loss = get_qe_loss(return_arr)[0]
            elif desc == 'qfirst':
                eval_loss = get_qe_loss(return_arr)[1]
            elif desc == 'qsecond':
                eval_loss = get_qe_loss(return_arr)[2]
            grads = torch.autograd.grad(eval_loss, model.parameters())
            loss_grad = torch.cat([grad.view(-1) for grad in grads]).data
            grads_list[descs.index(desc)].append(loss_grad.numpy())

    if len(variances) == len(descs):
        for var in variances:
            writer.write("{},".format(var))
        writer.write("\n")

    return grads_list


def aggregate_or_eval_grads(model, returns_arr, get_eval_loss, num_eval_grad_steps, grads_list, args, writer):
    variances = []
    if args.eval_grad_gae or args.eval_grad_qe:
        ranges = [0, 1, 2, 3, 10]
    else:
        ranges = [0, 0.5, 1, 2, 3, 10]

    for i, step in enumerate(ranges):
        eval_loss = get_eval_loss(returns_arr[i])
        grads = torch.autograd.grad(eval_loss, model.parameters())
        loss_grad = torch.cat([grad.view(-1) for grad in grads]).data
        if num_eval_grad_steps % args.eval_grad_freq == 0:
            total_grads = np.vstack(grads_list[i])
            variance = np.log(np.mean(np.var(total_grads, 0)))
            print("Log Gradient Variance for {}-step oracle model: {}".format(step, variance))
            variances.append(variance)
            grads_list[i] = []
        else:
            grads_list[i].append(loss_grad.numpy())

    if len(variances) == len(ranges):
        for var in variances:
            writer.write("{},".format(var))
        writer.write("\n")

    return grads_list

def trpo_step(model, get_loss, get_kl, max_kl, damping, get_grad=None):
    loss = get_loss()
    if get_grad:
      grads = torch.autograd.grad(loss, model.parameters())
      loss_grad = torch.cat([grad.view(-1) for grad in grads]).data
      print('Regular LOSS_GRAD_MSE: ', np.log(loss_grad.pow(2).mean()))
      loss_grad = get_grad()
      print('Factorized LOSS_GRAD_MSE: ', np.log(loss_grad.pow(2).mean()))
    else:
      grads = torch.autograd.grad(loss, model.parameters())
      loss_grad = torch.cat([grad.view(-1) for grad in grads]).data
      print('Regular LOSS_GRAD_MSE: ', np.log(loss_grad.pow(2).mean()))

    # Get grads wrt different losses
    # Store across epochs
    # Compute variances

    def Fvp(v):
        kl = get_kl()
        kl = kl.mean()

        grads = torch.autograd.grad(kl, model.parameters(), create_graph=True)
        flat_grad_kl = torch.cat([grad.view(-1) for grad in grads])

        kl_v = (flat_grad_kl * Variable(v)).sum()
        grads = torch.autograd.grad(kl_v, model.parameters())
        flat_grad_grad_kl = torch.cat([grad.contiguous().view(-1) for grad in grads]).data

        return flat_grad_grad_kl + v * damping

    stepdir = conjugate_gradients(Fvp, -loss_grad, 10)

    shs = 0.5 * (stepdir * Fvp(stepdir)).sum(0, keepdim=True)

    lm = torch.sqrt(shs / max_kl)
    fullstep = stepdir / lm[0]

    neggdotstepdir = (-loss_grad * stepdir).sum(0, keepdim=True)
    print(("lagrange multiplier:", lm[0], "grad_norm:", loss_grad.norm()))

    prev_params = get_flat_params_from(model)
    success, new_params = linesearch(model, get_loss, prev_params, fullstep,
                                     neggdotstepdir / lm[0])
    set_flat_params_to(model, new_params)

    return loss
