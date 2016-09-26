Beautiful Architecture Notes
===
> Oreilly

# Part I: On Architecture

## Chapter 1: What Is Architecture
> john klein, david weiss  

Central to beauty is conceptual integrity -- that is, a set of abstractions
and the rules for using them thoughout the system as simply as possible.

In all disiplines, architecture provides a means for solving a common problem:
assuring that a building, or bridge, or composition, or book, or network, or
system has certain properties and behaviours when it has been built.

A good system architecture exihibits conceptual integrity; that is, it comes
equipped with a set of design rules that aid in reducing complexity and that
can be used as guidande in detailed design and in system verification.

The architecture of a computer system we define as the minimal set of
properties that determine what programs will run and what results they will be.

Architecture is a game of trade-offs -- a decision that improves one the these
characteristics often diminishes another.

In wideruse, the term "architecture" always means "unchanging deep structure".

Structures provide ways to understand the system as sets of interacting
components.

We employ an architect to assure that the design (1) meets the needs; (2) has
conceptual integrity; (3) meets legal and safety requirements.

Architect must make many design decisions.

External behavioral descriptions show how the product will interface with its
users, other systems, and external devices, and should take the form of
requirements. Structural descriptions show how the product is devided into
parts and the relations between those parts. Internal behavioural descriptions
are needed to describe the interface between components.

Software architectures are often presented as layered hierarchies that tend
to commingle several different structures in one diagram.

Describing the structures of an architecture as a set of views, each of which
addresses different concerns, is now accepted as a standard architecture
practice.

Architecture is a part of the design of the system; it highlights some details
by abstracting away from others. Architecture is thus a subset of design.

The first concern of a software architect is not the functionality of the
system, but the quality concerns that needed to be satisfied.

Architecture decisions will have an impact on what kinds of changes can be
done easily and quickly and what changes will take time and be hard to do.

In the extreme case, there is no decomposition, and the system is developed as
a monolithic block of software.

Choosing one of the technologies is and architecture decision that will have
significant impact on your ability to meet cartain quality requirements.

We hope you understand by now that architecture decisions are important if your
system is going to meet its quality requirements, and what you want to pay
attention to the architecture and make these decisions intentionaly rather than
just "letting the architecture emerge".

The mind of a single user must comprehend a computer architecture. If a planned
architecture cannot be designed by a single mind, it cannot be comprehended by
one.

Fred Brooks said that conceptual integrity is the most important attribute of
an architecture.

* Functionality  
* Changeability  
* Performance  
* Capacity  
* Ecosystem  
* Modularity  
* Buildability  
* Producibility  
* Security  

Key structural decisions divide the product into components and define the
relationships among those components.

#### Summary of structures

* Information hiding structure: "is part of", "is contained in": changeability,
modularity, buildability  
* Uses structure: "uses": producibility, ecosystem  
* Process structure: "gives work to", "gets resources from", "shares resource
with", "contained in": performance, changeability, capacity  
* Data access structure: "has access to": security, ecosystem  

Searching the web for "software architecture review checklist" returns dozens
of chechlists.

Albert Einstein might say that beautiful architecture are as simple as possible,
but no simpler.

Our last example is the Unix system, which exhibits conceptual integrity, is
widely used, and has had great influence. The pipe and filters design is a
lovely abstraction that permits rapid construction of new applications.

> "Chapter1" was marked as read in Sept 2016.  

## Chapter 2: A Table of two systems: a modern-day software fable
TODO
