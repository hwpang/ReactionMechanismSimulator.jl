using Parameters
using SpecialFunctions
using LinearAlgebra

@inline function calcgibbs(ph::U,T::W) where {U<:IdealPhase,W<:Real}
    return getGibbs.(getfield.(ph.species,:thermo),T)
end
export calcgibbs

@inline function calcenthalpyinternalgibbs(ph::U,T::W,P::Z,V::Q) where {U<:IdealPhase,W,Z,Q<:Real}
    Hs = getEnthalpy.(getfield.(ph.species,:thermo),T)
    Us = Hs .- P*V
    Gs = Hs .- T.*getEntropy.(getfield.(ph.species,:thermo),T)
    return Hs,Us,Gs
end


@inline function calcenthalpyinternalgibbs(ph::IdealGas,T::W,P::Z,V::Q) where {W,Z,Q<:Real}
    Hs = getEnthalpy.(getfield.(ph.species,:thermo),T)
    Us = Hs .- R*T
    Gs = Hs .- T.*getEntropy.(getfield.(ph.species,:thermo),T)
    return Hs,Us,Gs
end

export calcenthalpyinternalgibbs

function makespcsvector(phase,spcdict)
    y0 = zeros(length(phase.species))
    spnames = [x.name for x in phase.species]
    for (key,val) in spcdict
        if key == "T"
            continue
        elseif key == "P"
            continue
        elseif key == "V"
            continue
        else
            ind = findfirst(isequal(key),spnames)
            @assert typeof(ind)<: Integer  "$key not found in species list: $spnames"
            y0[ind] = val
        end
    end
end

export makespcsvector

@inline function getkf(rxn::ElementaryReaction,ph::U,T::W1,P::W2,C::W3,ns::Q,V::W4) where {U<:AbstractPhase,W1,W2,W3,W4<:Real,Q<:AbstractArray}
    if isdefined(rxn.kinetics,:efficiencies) && length(rxn.kinetics.efficiencies) > 0
        @views @inbounds @fastmath C += sum([ns[i]*val for (i,val) in rxn.kinetics.efficiencies])/V
    end
    return rxn.kinetics(T=T,P=P,C=C)
end
export getkf

"""
Calculates the diffusion limited rate coefficient
for 1 spc returns Inf
for 2 spc calculates using the Smolchowski equation
for >2 spc calculates using the Generalized Smolchowski equation
Equations from Flegg 2016
"""
@inline function getDiffusiveRate(spcs::Q,diffs::Array{W,1}) where {Q<:AbstractArray,W<:Real}
    if length(spcs) == 1
        return Inf
    elseif length(spcs) == 2
        @fastmath @inbounds kf = 4.0*Base.pi*(diffs[spcs[1].index]+diffs[spcs[2].index])*(spcs[1].radius+spcs[2].radius)*Na
    else
        @views @inbounds diffusivity = diffs[getfield.(spcs,:index)]
        N = length(spcs)
        @fastmath a = (3.0*length(spcs)-5.0)/2.0
        @fastmath Dinv = 1.0./st.diffusivity
        @fastmath Dbar = 1.0./reverse(cumsum(Dinv))
        @fastmath Dhat = st.diffusivity .+ Dbar
        @fastmath @inbounds deltaN = sum(Dinv)/sum(sum([[1.0/(st.diffusivity[i]*st.diffusivity[m]) for m in 1:N-1 if i>m] for i in 2:N]))
        @views @fastmath @inbounds kf = prod(Dhat[2:end].^1.5)*4*Base.pi^(a+1)/gamma(a)*(sum(getfield.(spcs,:radius))/sqrt(deltaN))^(2*a)*Na^(N-1)
    end
    return kf
end
export getDiffusiveRate

@inline function getKc(rxn::ElementaryReaction,ph::U,T::Z,Gs::Array{Q,1}) where {U<:AbstractPhase,Q,Z<:Real}
    Nreact = length(rxn.reactantinds)
    Nprod = length(rxn.productinds)
    dGrxn = 0.0
    if Nreact == 1
        @fastmath @inbounds dGrxn -= Gs[rxn.reactantinds[1]]
    elseif Nreact == 2
        @fastmath @inbounds dGrxn -= Gs[rxn.reactantinds[1]]+Gs[rxn.reactantinds[2]]
    elseif Nreact == 3
        @fastmath @inbounds dGrxn -= Gs[rxn.reactantinds[1]]+Gs[rxn.reactantinds[2]]+Gs[rxn.reactantinds[3]]
    end
    if Nprod == 1
        @fastmath @inbounds dGrxn += Gs[rxn.productinds[1]]
    elseif Nprod == 2
        @fastmath @inbounds dGrxn += Gs[rxn.productinds[1]]+Gs[rxn.productinds[2]]
    elseif Nprod == 3
        @fastmath @inbounds dGrxn += Gs[rxn.productinds[1]]+Gs[rxn.productinds[2]]+Gs[rxn.productinds[3]]
    end
    return @inbounds @fastmath exp(-dGrxn/(R*T))*(1.0e5/(R*T))^(Nprod-Nreact)
end
export getKc

"""
Calculates the forward and reverse rate coefficients for a given reaction, phase and state
Maintains diffusion limitations if the phase has diffusionlimited=true
"""
@inline function getkfkrev(rxn::ElementaryReaction,ph::U,T::W1,P::W2,C::W3,N::W4,ns::Q1,Gs::Q2,diffs::Q3,V::W5) where {U<:AbstractPhase,W5,W4,W1,W2,W3<:Real,Q1,Q2,Q3<:AbstractArray}
    kf = getkf(rxn,ph,T,P,C,ns,V)
    Kc = getKc(rxn,ph,T,Gs)
    @fastmath krev = kf/Kc
    if ph.diffusionlimited
        if length(rxn.reactants) == 1
            if length(rxn.products) > 1
                krevdiff = getDiffusiveRate(rxn.products,diffs)
                @fastmath krev = krev*krevdiff/(krev+krevdiff)
                @fastmath kf = Kc*krev
            end
        elseif length(rxn.products) == 1
            kfdiff = getDiffusiveRate(rxn.reactants,diffs)
            @fastmath kf = kf*kfdiff/(kf+kfdiff)
            @fastmath krev = kf/Kc
        elseif length(rxn.products) == length(rxn.reactants)
            kfdiff = getDiffusiveRate(rxn.reactants,diffs)
            krevdiff = getDiffusiveRate(rxn.products,diffs)
            @fastmath kff = kf*kfdiff/(kf+kfdiff)
            @fastmath krevr = krev*krevdiff/(krev+krevdiff)
            @fastmath kfr = Kc*krevr
            if kff > kfr
                kf = kfr
                krev = krevr
            else
                kf = kff
                @fastmath krev = kf/Kc
            end
        end
    end
    return (kf,krev)::Tuple{W3,W3}
end
export getkfkrev

@inline function getkfkrevs(;phase::U,T::W1,P::W2,C::W3,N::W4,ns::Q1,Gs::Q2,diffs::Q3,V::W5) where {U<:AbstractPhase,W5<:Real,W1<:Real,W2<:Real,W3<:Real,W4<:Real, Q1<:AbstractArray,Q2<:AbstractArray,Q3<:AbstractArray}
    len = length(phase.reactions)
    kf = zeros(typeof(N),len)
    krev = zeros(typeof(N),len)
    @simd for i = 1:len
       @fastmath @inbounds kf[i],krev[i] = getkfkrev(phase.reactions[i],phase,T,P,C,N,ns,Gs,diffs,V)
    end
    return kf,krev
end
export getkfkrevs
