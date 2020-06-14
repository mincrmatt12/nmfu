pipeline {
	agent {
		label "docker && linux"
	}
	stages {
		stage ('Build') {
			agent {
				docker { 
					image 'python:3' 
					label "docker && linux"
				}
			}
			environment {
				TAG_BUILD = """${sh(returnStdout: true, script: 'bash -c "[[ \\$TAG_NAME ]] && true || echo dev0+git-${GIT_COMMIT}"').trim()}"""
			}
			steps {
				sh "python setup.py egg_info --tag-build ${TAG_BUILD} sdist bdist_wheel"
				archiveArtifacts artifacts: 'dist/*'
			}
		}
		stage ('Test') {
			agent {
				dockerfile {
					label "docker && linux"
					filename 'Dockerfile.build'
				}
			}
			steps {
				sh "pytest --junit-xml=junit.xml || true"
				junit 'junit.xml'
			}
		}
		stage ('Snapify') {
			agent {
				label "scala"
			}
			steps {
				sh "snapcraft --use-lxd"
				archiveArtifacts artifacts: '*.snap'
			}
		}
	}
}
