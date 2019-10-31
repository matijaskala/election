struct Tensor
    data::Array{Int64,3}
    votes::Int64
    Tensor(C) = new(fill(0, (C, C, 5)), 0)
end

function add_ballot(tensor, ballot)
    dim = size(tensor.data)
    for i in 1:dim[1]
        for j in 1::dim[2]
            for k in 1::dim[3]
                if ballot[i] >= k && ballot[j] >= k
                    tensor.data[i, j, k] += 1
                end
            end
        end
    end
end

function get_score(tensor, candidate, already_elected, cutoff)
    d = 1.0
    for a in already_elected
        if tensor.data[a, a, cutoff] == 0
            return 0.0
        end
        d += tensor.data[candidate, a, cutoff] / tensor.data[a, a, cutoff]
    end
    return tensor.data[candidate, candidate, cutoff] / d
end

function get_cutoff(tensor, Q, already_elected)
    dim = size(tensor.data)
    for cutoff in dim[3]:-1:2
        for a in 1:dim[1]
            if a in already_elected
                continue
            end
            if get_score(tensor, a, already_elected, cutoff-1) >= Q
                return cutoff
            end
        end
    end
    return 1
end

function get_next_winner(tensor, Q, already_elected)
    dim = size(tensor.data)
    idx = Int32[]
    for cutoff in get_cutoff(tensor, Q, already_elected):dim[3]
        if size(idx)[1] == 0
            for a in 1:dim[1]
                if a in already_elected
                    continue
                end
                if cutoff == 1 || get_score(tensor, a, already_elected, cutoff-1) >= Q
                    append!(idx, a)
                end
            end
        elseif size(idx)[1] == 1
            break
        end
        val = -1.0
        idx2 = Int32[]
        for a in idx
            cur = get_score(tensor, a, already_elected, cutoff)
            if cur == val
                append!(idx2, a)
            elseif cur > val
                val = cur
                idx2 = Int32[]
                append!(idx2, a)
            end
        end
        idx = idx2
    end
    return idx[1]
end

function get_winners(tensor, W)
    r = Int32[]
    for n in 1:W
        append!(r, get_next_winner(tensor, tensor.votes / W, r))
    end
end
