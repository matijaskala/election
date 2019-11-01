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

function get_score(tensor, candidate1, candidate2, already_elected, cutoff)
    d1 = 1.0
    d2 = 1.0
    votes1 = tensor.data[candidate1, candidate1, cutoff]
    votes2 = tensor.data[candidate2, candidate2, cutoff]
    if votes1 == 0 || votes2 == 0
        return 0.0
    end
    for a in already_elected
        a_votes = tensor.data[a, a, cutoff]
        if a_votes == 0
            return 0.0
        end
        d1 += tensor.data[candidate1, a, cutoff] / a_votes
        d2 += tensor.data[candidate2, a, cutoff] / a_votes
    end
    invscore1 = d1 / votes1
    invscore2 = d2 / votes2
    invscoreX = tensor.data[candidate1, candidate2, cutoff] / (votes1 * votes2)
    return 2 / (invscore1 + invscore2 + invscoreX)
end

function get_cutoff(tensor, Q, already_elected, are_different_genders)
    dim = size(tensor.data)
    for cutoff in dim[3]:-1:2
        for a in 1:dim[1]
            for b in 1:a-1
                if !are_different_genders(a, b)
                    continue
                elseif a in already_elected || b in already_elected
                    continue
                elseif get_score(tensor, a, b, already_elected, cutoff-1) >= Q
                    return cutoff
                end
            end
        end
    end
    return 1
end

function get_next_winners(tensor, Q, already_elected, are_different_genders)
    dim = size(tensor.data)
    idx = Tuple{Int32, Int32}[]
    for cutoff in get_cutoff(tensor, Q, already_elected, are_different_genders):dim[3]
        if size(idx)[1] == 0
            for a in 1:dim[1]
                for b in 1:a-1
                    if !are_different_genders(a, b)
                        continue
                    elseif a in already_elected || b in already_elected
                        continue
                    elseif cutoff == 1 || get_score(tensor, a, b, already_elected, cutoff-1) >= Q
                        push!(idx, (a, b))
                    end
                end
            end
        elseif size(idx)[1] == 1
            break
        end
        val = -1.0
        idx2 = Tuple{Int32, Int32}[]
        for (a, b) in idx
            cur = get_score(tensor, a, b, already_elected, cutoff)
            if cur == val
                push!(idx2, (a, b))
            elseif cur > val
                val = cur
                idx2 = Tuple{Int32, Int32}[]
                push!(idx2, (a, b))
            end
        end
        idx = idx2
    end
    return idx[1]
end

function get_winners(tensor, W, are_different_genders)
    r = Int32[]
    while size(r)[1] < W
        append!(r, get_next_winners(tensor, tensor.votes / W, r, are_different_genders))
    end
end
