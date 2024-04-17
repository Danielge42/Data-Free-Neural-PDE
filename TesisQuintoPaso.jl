using Flux, Statistics

R=2
L=3
mu=3
p0=10
pf=3
dp=(p0-pf)

dim=32
lr=0.001


w1=rand(dim,1)/100
b1=rand(dim)/100
layer1(x)=w1*x .+b1

w2=rand(1,dim)/100
b2=rand(1)/100
layer2(x)= w2*x .+b2

p= x-> layer2(tanh.(layer1(x)))[1]
a=0
layer1(1)-layer1(1.5)
v(x)=(x-R)*(x+R)*p(x) 
p(a)
h=0.01
function loss()
    #mean(abs2(((v(x+h)-v(x))/h) + 2 ) for x in -R:0.01:R)
    mean(abs2(v(R))+abs2(v(-R))+abs2(x*(v(x+h)-2*v(x)+v(x-h))/h^2 + ((v(x+h)-v(x))/h) + x*dp/(L*mu)) for x in -R+h:h:R-h) 
end
function gradiente(w1,b1,w2,b2)
    
    #dw2

    dw2= mean(2*(x*(v(x+h)-2*v(x)+v(x-h))/h^2 + (v(x+h)-v(x))/h + x*dp/(L*mu))*
    (x*((x+h-R)*(x+h+R)*tanh.(layer1(x+h)) -2*(x-R)*(x+R)*tanh.(layer1(x))
     +(x-h-R)*(x-h+R)*tanh.(layer1(x-h)))/h^2 
    + ((x+h-R)*(x+h+R)*tanh.(layer1(x+h))-(x-R)*(x+R)*tanh.(layer1(x)))/h)
        for x in -R:h:R)

     
    #db2

    db2=mean(2*(x*(v(x+h)-2*v(x)+v(x-h))/h^2 + (v(x+h)-v(x))/h + x*dp/(L*mu))*
                (x*((x+h-R)*(x+h+R)-2*(x+R)*(x-R)+(x-h-R)*(x-h+R))/h^2
                +((x+h+R)*(x+h-R)-(x+R)*(x-R))/h ) for x in -R:h:R)


    w2t=transpose(w2)
    #dw1
    dw1= mean(2*(x*(v(x+h)-2*v(x)+v(x-h))/h^2 + (v(x+h)-v(x))/h + x*dp/(L*mu))*
    
            (x*((x+h+R)*(x+h-R)*(x+h)*w2t .*((sech.(layer1(x+h)))).^2 
             -2*(x-R)*(x+R)*x*w2t .*((sech.(layer1(x)))).^2
            +(x-h+R)*(x-h-R)*(x-h)*w2t .*((sech.(layer1(x-h))) ).^2)/h^2 
            +((x+h-R)*(x+h+R)*(x+h)*w2t .*((sech.(layer1(x+h)))).^2
             -(x-R)*(x+R)*x*w2t .*(sech.(layer1(x))).^2)/h)
             for x in -R:h:R)
    #db1
    db1= mean(2*(x*(v(x+h)-2*v(x)+v(x-h))/h^2 + (v(x+h)-v(x))/h + x*dp/(L*mu))
    
            *(x*((x+h+R)*(x+h-R)*w2t .*((sech.(layer1(x+h)))).^2 
             -2*(x-R)*(x+R)*w2t .*((sech.(layer1(x)))).^2
            +(x-h+R)*(x-h-R)*w2t .*((sech.(layer1(x-h))) ).^2)/h^2 
            +((x+h-R)*(x+h+R)*w2t .*((sech.(layer1(x+h)))).^2
             -(x-R)*(x+R)*w2t .*(sech.(layer1(x))).^2 )/h)
             for x in -R:h:R)


    return dw1,db1,transpose(dw2),db2
end


gradiente(w1,b1,w2,b2)[3]
g=gradient(()->loss(), Flux.params(w1, b1,w2,b2))
g[w2]
epochs=500
for i in 0:1:epochs
    gs = gradiente(w1, b1,w2,b2)
    w1n=gs[1]
    w1 .-= lr .*w1n
    w2n=gs[3]
    w2 .-= lr .*w2n
    b1n=gs[2]
    b1 .-= lr*b1n
    b2n=gs[4]
    b2 .-=lr*b2n
    if i %100==0
        display(loss())
    end

end

using Plots
u= r -> dp/(4*mu*L)*R^2*(1-(r/R)^2)
rsr = -R:0.1:R

rsp = -R:0.1:R
plot(rsp,v.(rsp),label="NN")
plot!(rsr,u.(rsr),label="real")
plot!(rsp,v.(rsp)-u.(rsr),label="resta")