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
        if tensor.data[a, a, cutoff]
            return 0.0
        end
        d += tensor.data[candidate, a, cutoff] / tensor.data[a, a, cutoff]
    end
    return tensor.data[candidate, candidate, cutoff]
end

function get_cutoff(tensor, Q, already_elected)
    for cutoff in 5:-1:2
        for a in already_elected
            if get_score(tensor, a, already_elected, cutoff-1) >= Q
                return cutoff
            end
        end
    end
    return 1
end

function get_next_winner(tensor, Q, already_elected)
    idx = zeros(Int32, 0)
    for cutoff in get_cutoff(tensor, Q, already_elected):5
        if size(idx) == 0
            for a in 1:size(tensor.data)[1]
                append!(idx, a)
            end
        elseif size(idx) == 1
            break
        end
        val = -1.0
        idx2 = zeros(Int32, 0)
        for a in idx
            cur = get_score(tensor, a, already_elected, cutoff)
            if cur == val
                append!(idx2, a)
            elseif cur > val
                val = cur
                idx2 = zeros(Int32, 0)
                append!(idx2, a)
            end
        end
        idx = idx2
    end
    return idx[1]
end

function get_winners(tensor, W)
    r = zeros(Int32, 0)
    for n in 1:W
        append!(r, get_next_winner(tensor, tensor.votes / W, r))
    end
end