using Symbolics
using ModelingToolkit

function getmapping(phase::T,qssnames::A1,isomergroups::A2) where {T<:AbstractPhase,A1<:AbstractArray,A2<:AbstractArray}
    
    spcnames = getfield.(phase.species,:name)
    qssindexes = [index for (index, name) in enumerate(spcnames) if name in qssnames]
    
    lumpedgroupmapping = [Dict{Int64,Float64}() for group in isomergroups]
    lumpedindexes = Array{Int64,1}()
    for (i,group) in enumerate(isomergroups)
        for (spcname, weight) in group
            index = findfirst(x->x==spcname, spcnames)
            lumpedgroupmapping[i][index] = weight
            push!(lumpedindexes,index)
        end
    end
    reducedindexes = [index for index in 1:length(spcnames) if !(index in qssindexes) && !(index in lumpedindexes)]
    return qssindexes,lumpedindexes,reducedindexes,lumpedgroupmapping
end
export getmapping

function generateqsscmapping(phase::T,qssnames::A1,isomergroups::A2;saveqssc::Bool=false,outputname::String="qssc.jl") where {T<:AbstractPhase,A1<:AbstractArray,A2<:AbstractArray}

    qssindexes,lumpedindexes,reducedindexes,lumpedgroupmapping = getmapping(phase, qssnames, isomergroups)
    
    @parameters symkf[1:length(phase.reactions)] symkrev[1:length(phase.reactions)] symV
    @variables symdc[1:length(phase.species)] symc[1:length(phase.species)]
    
    symdc .= Num(0);

    addreactionratecontributions!(symdc,phase.rxnarray,symc,symkf,symkrev)
    
    eqs = 0 .~ symdc[qssindexes];

    nonlinearterms = Dict([])
    for i in symc[qssindexes]
        nonlinearterms[i*i]=0 #x^2=>0
    end

    for ind in 1:length(eqs)
        if !(Symbolics.isaffine(eqs[ind].rhs,symc[qssindexes]))
            eqs[ind] = 0~substitute(eqs[ind].rhs,nonlinearterms)
            filter!(x->length(get_variables(x.first,symc[qssindexes])) < 2,eqs[ind].rhs.dict) #x*y
            eqs[ind].rhs.sorted_args_cache[] = nothing
            @assert Symbolics.isaffine(eqs[ind].rhs,symc[qssindexes]) == true
        end
    end
    
    qsscsolved = Symbolics.expand(Symbolics.solve_for(eqs,symc[qssindexes];simplify=false))
    
    symqssc! = build_function(qsscsolved, symc, symkf, symkrev)[2]

    if saveqssc
        write(outputname,string(symqssc!));
    end
    
    return qssindexes,lumpedindexes,reducedindexes,lumpedgroupmapping,eval(symqssc!)
    
end
export generateqsscmapping

